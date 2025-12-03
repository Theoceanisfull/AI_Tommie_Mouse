import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import deque

# Allow imports from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.mouse_env import MouseMazeEnvLegacy  # noqa: E402
from src.agents import ImitationAgent  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "data", "3131")
META_PATH = os.path.join(DATA_DIR, "maze_metadata.csv")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
MODEL_PATH = os.path.join(ROOT, "trained_models", "imitation_multi.pth")
EPOCHS = 10 # Increased 
BATCH_SIZE = 64 # Decreased Batch Size
STEP_MULTIPLIER = 2.0  # allow more steps during eval to reduce timeouts
OBS_NOISE_STD = 0.001 # small Gaussian noise for regularization; changed from 0.01 -> 

def direction_to_action(prev, curr):
    dr = curr[0] - prev[0]
    dc = curr[1] - prev[1]
    mapping = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    return mapping.get((dr, dc), None)


def collect_dataset(df):
    observations = []
    actions = []
    for _, row in df.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        path_file = row.get("path_file", None)
        if not isinstance(path_file, str):
            continue
        full_path = os.path.join(DATA_DIR, path_file)
        if not os.path.exists(full_path):
            continue
        path = np.load(full_path)
        # Validate start/goal markers
        if 2 not in maze or 3 not in maze:
            raise ValueError(f"Maze {row['filename']} missing start (2) or goal (3) markers.")
        env = MouseMazeEnvLegacy(maze, optimal_path=path, render_mode=None)
        obs, _ = env.reset()
        path_seq = path.tolist()
        for i in range(1, len(path_seq)):
            prev = tuple(path_seq[i - 1])
            curr = tuple(path_seq[i])
            action = direction_to_action(prev, curr)
            if action is None:
                continue
            observations.append(obs)
            obs, _, _, _, _ = env.step(action)
            actions.append(action)
        env.close()
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def eval_split(agent, df):
    records = []
    for _, row in df.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        path_file = row.get("path_file", None)
        # Derive optimal steps from path file when available; fall back to metadata
        optimal_steps = None
        if isinstance(path_file, str) and os.path.exists(os.path.join(DATA_DIR, path_file)):
            try:
                opt_path = np.load(os.path.join(DATA_DIR, path_file))
                optimal_steps = max(1, len(opt_path) - 1)
            except Exception:
                optimal_steps = None
        if optimal_steps is None:
            meta_opt = row.get("optimal_steps", None)
            if meta_opt is not None and meta_opt > 0:
                optimal_steps = meta_opt
        if 2 not in maze or 3 not in maze:
            raise ValueError(f"Maze {row['filename']} missing start (2) or goal (3) markers.")
        env = MouseMazeEnvLegacy(maze, optimal_path=None, render_mode=None)
        obs, _ = env.reset()
        agent.reset_state()
        steps = 0
        success = False
        truncated_flag = False
        max_steps = int(env.max_steps * STEP_MULTIPLIER)
        while steps < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated:
                success = True
                break
            if truncated:
                truncated_flag = True
                break
        env.close()
        # Only compute ratio if we have a valid optimal and the agent actually finished
        if success and optimal_steps is not None and optimal_steps > 0:
            step_ratio = steps / optimal_steps
        else:
            step_ratio = np.nan
        records.append(
            {
                "filename": row["filename"],
                "complexity": row.get("complexity", "unknown"),
                "success": success,
                "steps": steps,
                "step_ratio": step_ratio,
                "truncated": truncated_flag,
            }
        )
    return pd.DataFrame(records)


def train_imitation_multi():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")

    meta = pd.read_csv(META_PATH)
    if "split" not in meta.columns:
        meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(meta) * 0.9)
        meta["split"] = "train"
        meta.loc[split_idx:, "split"] = "test"
        meta.to_csv(META_PATH, index=False)

    # Compute a path length column (prefer metadata, fallback to path file length)
    if "path_len" not in meta.columns:
        meta["path_len"] = np.nan
    for i, row in meta.iterrows():
        if pd.notna(meta.at[i, "path_len"]) and meta.at[i, "path_len"] > 0:
            continue
        if "optimal_steps" in meta.columns and pd.notna(row.get("optimal_steps", np.nan)):
            meta.at[i, "path_len"] = row["optimal_steps"]
            continue
        path_file = row.get("path_file", None)
        if isinstance(path_file, str):
            pfile = os.path.join(DATA_DIR, path_file)
            if os.path.exists(pfile):
                try:
                    p = np.load(pfile)
                    meta.at[i, "path_len"] = max(1, len(p) - 1)
                except Exception:
                    pass
    meta["path_len"] = meta["path_len"].fillna(meta["path_len"].median())

    train_df = meta[meta["split"] == "train"]
    test_df = meta[meta["split"] == "test"]

    print(f"Collected {len(train_df)} train mazes, {len(test_df)} test mazes from {DATA_DIR}")

    # Simple reverse curriculum: train on shorter paths first (sorted by path_len)
    train_df = train_df.sort_values("path_len")

    obs_data, act_data = collect_dataset(train_df)
    obs_dim = obs_data.shape[1]
    n_actions = 4

    agent = ImitationAgent(obs_dim, n_actions)

    losses = []
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(obs_data))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(obs_data), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            obs_batch = obs_data[idx]
            if OBS_NOISE_STD > 0:
                obs_batch = obs_batch + np.random.normal(0, OBS_NOISE_STD, obs_batch.shape).astype(np.float32)
            loss = agent.train_step(obs_batch, act_data[idx])
            epoch_loss += loss
            batches += 1
        avg_loss = epoch_loss / max(1, batches)
        losses.append(avg_loss)
        agent.step_scheduler(avg_loss)
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            print(f"[Multi] Epoch {epoch+1}/{EPOCHS} loss: {avg_loss:.4f}")

    torch.save(agent.state_dict(), MODEL_PATH)
    print(f"Saved imitation multi model to {MODEL_PATH}")

    # Training loss plot
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Imitation Multi Training Loss")
    plt.grid(True)
    loss_path = os.path.join(METRICS_DIR, "imitation_multi_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved loss plot to {loss_path}")

    # Evaluate on train/test
    agent.eval()
    train_eval = eval_split(agent, train_df)
    test_eval = eval_split(agent, test_df)

    # Metrics
    train_success = train_eval["success"].mean()
    test_success = test_eval["success"].mean()
    train_ratios = train_eval["step_ratio"].dropna()
    test_ratios = test_eval["step_ratio"].dropna()
    train_mean = train_ratios.mean()
    test_mean = test_ratios.mean()
    train_median = train_ratios.median()
    test_median = test_ratios.median()

    print(f"[Multi Eval] train_success={train_success:.3f}, test_success={test_success:.3f}")
    print(f"[Multi Eval] step_ratio mean (train/test) = {train_mean:.2f}/{test_mean:.2f}")
    print(f"[Multi Eval] step_ratio median (train/test) = {train_median:.2f}/{test_median:.2f}")

    # Save eval CSVs
    train_eval.to_csv(os.path.join(METRICS_DIR, "imitation_multi_train_eval.csv"), index=False)
    test_eval.to_csv(os.path.join(METRICS_DIR, "imitation_multi_test_eval.csv"), index=False)

    # Plots for success rate and step ratio
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Train", "Test"], [train_success, test_success])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Success Rate")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    if len(train_ratios) == 0 and len(test_ratios) == 0:
        axes[1].text(0.5, 0.5, "No successful episodes to plot", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        box_data = [train_ratios if len(train_ratios) > 0 else [np.nan],
                    test_ratios if len(test_ratios) > 0 else [np.nan]]
        axes[1].boxplot(box_data, labels=["Train", "Test"], showmeans=True)
        axes[1].set_ylabel("Step Ratio (steps/optimal)")
        axes[1].set_title("Step Ratio Distribution")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    eval_plot = os.path.join(METRICS_DIR, "imitation_multi_eval.png")
    plt.savefig(eval_plot)
    plt.close()
    print(f"Saved eval plot to {eval_plot}")


if __name__ == "__main__":
    train_imitation_multi()
