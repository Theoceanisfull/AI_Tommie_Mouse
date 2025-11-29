import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

from src.maze_generator import MazeGenerator
from src.mouse_env import SimpleMazeEnv, MouseMazeEnv, MouseMazeEnvLegacy
from src.agents import (
    get_ppo_model,
    get_dqn_model,
    EvolutionaryTrainer,
    ImitationAgent,
    train_q_learning,
    QLearningAgent,
)

DATA_DIR_BASE = "src/data"  # per-difficulty folders (1111, 2121, 3131)
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def ea_checkpoint_path(env_name: str):
    return os.path.join(MODEL_DIR, f"ea_{env_name}.pth")


def make_env(maze, optimal_path=None, render_mode=None, env_name="simple"):
    if env_name == "mouse":
        # Use legacy straight-move FoV env (closest to prior imitation setup)
        return MouseMazeEnvLegacy(maze, optimal_path=optimal_path, render_mode=render_mode)
    return SimpleMazeEnv(maze, render_mode=render_mode)

def load_random_maze(complexity="Easy", split="train", maze_type=None):
    """Loads a random maze of specific difficulty (and optional type) from per-difficulty dataset."""
    subdir_map = {"Easy": "1111", "Medium": "2121", "Hard": "3131"}
    subdir = subdir_map.get(complexity, "2121")
    data_dir = os.path.join(DATA_DIR_BASE, subdir)
    meta_path = os.path.join(data_dir, "maze_metadata.csv")
    if not os.path.exists(meta_path):
        print(f"No dataset found at {meta_path}. Run generation first.")
        return None

    df = pd.read_csv(meta_path)
    filtered = df[df["difficulty"] == complexity] if "difficulty" in df.columns else df
    if maze_type and "type" in df.columns:
        filtered = filtered[filtered["type"] == maze_type]
    if "split" in filtered.columns and split:
        filtered = filtered[filtered["split"] == split]
    if filtered.empty:
        return None

    row = filtered.sample(1).iloc[0]
    maze = np.load(os.path.join(data_dir, row["filename"]))
    path = None
    if "path_file" in row and isinstance(row["path_file"], str):
        path_path = os.path.join(data_dir, row["path_file"])
        if os.path.exists(path_path):
            path = np.load(path_path)
    return maze, row.get("optimal_steps", -1), path


def _direction_to_action(prev, curr):
    dr = curr[0] - prev[0]
    dc = curr[1] - prev[1]
    mapping = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    return mapping.get((dr, dc), None)


def build_imitation_dataset(limit=200, split="train", env_name="mouse"):
    """Collect (obs, action) pairs by walking optimal paths."""
    meta_path = os.path.join(DATA_DIR, "maze_metadata.csv")
    if not os.path.exists(meta_path):
        print("No dataset found. Run generation first.")
        return None, None

    df = pd.read_csv(meta_path)
    if "split" in df.columns:
        df = df[df["split"] == split]
    if df.empty:
        print("No mazes available for imitation dataset.")
        return None, None

    # Subsample for quick training
    df = df.sample(min(limit, len(df)), random_state=0)

    observations = []
    actions = []

    for _, row in df.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        path = None
        if "path_file" in row and isinstance(row["path_file"], str):
            path_path = os.path.join(DATA_DIR, row["path_file"])
            if os.path.exists(path_path):
                path = np.load(path_path)

        env = make_env(maze, optimal_path=path, env_name=env_name)
        path_seq = path.tolist() if path is not None else env._compute_optimal_path()[0]

        if path_seq is None or len(path_seq) < 2:
            continue

        obs, _ = env.reset()
        for i in range(1, len(path_seq)):
            prev = tuple(path_seq[i - 1])
            curr = tuple(path_seq[i])
            action = _direction_to_action(prev, curr)
            if action is None:
                continue
            observations.append(obs)
            obs, _, _, _, _ = env.step(action)
            actions.append(action)

    if not observations:
        print("No imitation data collected.")
        return None, None

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def plot_training_curves(output_path, rewards=None, success_curve=None, step_ratio_curve=None, losses=None, title="Training progress"):
    rows = 0
    if rewards is not None:
        rows += 1
    if success_curve is not None or step_ratio_curve is not None:
        rows += 1
    if losses is not None:
        rows += 1
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]
    idx = 0
    if rewards is not None:
        axes[idx].plot(rewards, label="Episode reward")
        axes[idx].set_ylabel("Reward")
        axes[idx].legend()
        idx += 1
    if success_curve is not None or step_ratio_curve is not None:
        if success_curve is not None:
            axes[idx].plot(success_curve, label="Cumulative success rate")
        if step_ratio_curve is not None:
            axes[idx].plot(step_ratio_curve, label="Cumulative step ratio")
        axes[idx].set_ylabel("Performance")
        axes[idx].legend()
        idx += 1
    if losses is not None:
        axes[idx].plot(losses, label="Loss")
        axes[idx].set_ylabel("Loss")
        axes[idx].legend()
    axes[-1].set_xlabel("Episodes / Epochs")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def evaluate_policy(algo, model, env, optimal_steps, episodes=10):
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        if algo == "ea" and hasattr(model, "reset_state"):
            model.reset_state()
        steps = 0
        done = False
        success = False
        while not done and steps < env.max_steps:
            if algo in ["ppo", "dqn", "imitation"]:
                action, _ = model.predict(obs, deterministic=True)
            elif algo == "ea":
                action = model.select_action(obs)
            elif algo == "qlearn":
                pos = getattr(env, "agent_pos", None)
                action = model.act(obs, explore=False, pos=pos)
            else:
                action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                success = True
        step_ratio = steps / max(1, optimal_steps)
        results.append((success, step_ratio))
    if not results:
        return 0.0, np.inf
    successes = [int(s[0]) for s in results]
    ratios = [s[1] for s in results]
    return float(np.mean(successes)), float(np.mean(ratios))


def load_imitation(env, env_name):
    obs_dim = int(np.prod(env.observation_space.shape))
    model = ImitationAgent(obs_dim, env.action_space.n)
    ckpt_path = f"{MODEL_DIR}/imitation_{env_name}.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    return model


def evaluate_imitation(difficulty="Medium", env_name="mouse", runs=50):
    """
    Evaluate imitation agent on random test mazes without rendering.
    Saves a simple plot to models/imitation_eval.png.
    """
    successes = []
    step_ratios = []
    for i in range(runs):
        maze_bundle = load_random_maze(difficulty, split="test")
        if maze_bundle is None:
            print("No maze available to evaluate. Generate data first.")
            return
        maze, optimal_steps, path = maze_bundle
        env = make_env(maze, optimal_path=path, render_mode=None, env_name=env_name)
        model = load_imitation(env, env_name)
        if hasattr(model, "reset_state"):
            model.reset_state()
        obs, _ = env.reset()
        done = False
        steps = 0
        success = False
        while not done and steps < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                success = True
        successes.append(1 if success else 0)
        step_ratios.append(steps / max(1, optimal_steps))
        env.close()

    success_rate = float(np.mean(successes))
    avg_step_ratio = float(np.mean(step_ratios))
    print(f"Imitation eval on {runs} runs [{difficulty}]: success_rate={success_rate:.2f}, avg_step_ratio={avg_step_ratio:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(np.cumsum(successes) / np.arange(1, len(successes) + 1))
    axes[0].set_ylabel("Cumulative Success Rate")
    axes[0].grid(True)

    axes[1].plot(np.cumsum(step_ratios) / np.arange(1, len(step_ratios) + 1))
    axes[1].set_ylabel("Cumulative Step Ratio")
    axes[1].set_xlabel("Episodes")
    axes[1].grid(True)

    out_path = os.path.join(MODEL_DIR, "imitation_eval.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved eval plot to {out_path}")

def train(algo, episodes=1000, env_name="simple", difficulty="Medium", maze_type=None):
    print(f"Training {algo}...")
    
    maze_bundle = load_random_maze(difficulty, split="train", maze_type=maze_type)
    if maze_bundle is None:
        return
    maze, optimal, path = maze_bundle
    eval_bundle = load_random_maze(difficulty, split="test", maze_type=maze_type) or maze_bundle

    env = make_env(maze, optimal_path=path, env_name=env_name)
    env = Monitor(env)

    if algo == "ppo":
        model = get_ppo_model(env)
        model.learn(total_timesteps=episodes * 20) # approx steps
        model.save(f"{MODEL_DIR}/ppo_mouse")
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = make_env(eval_maze, optimal_path=eval_path, env_name=env_name)
        success_rate, avg_ratio = evaluate_policy("ppo", model, eval_env, eval_optimal)
        print(f"PPO eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")
    
    elif algo == "dqn":
        model = get_dqn_model(env)
        model.learn(total_timesteps=episodes * 20)
        model.save(f"{MODEL_DIR}/dqn_mouse")
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = make_env(eval_maze, optimal_path=eval_path, env_name=env_name)
        success_rate, avg_ratio = evaluate_policy("dqn", model, eval_env, eval_optimal)
        print(f"DQN eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")
        
    elif algo == "ea":
        env_kwargs = {}
        # Build a small curriculum batch of Easy mazes for EA to train against
        curriculum = [maze]
        for _ in range(3):
            extra = load_random_maze("Easy", split="train")
            if extra is not None:
                curriculum.append(extra[0])

        trainer = EvolutionaryTrainer(
            curriculum,
            generations=30,
            population_size=100,
            env_cls=SimpleMazeEnv,
            env_kwargs=env_kwargs,
        )
        best_agent, _ = trainer.train()
        torch.save(
            {
                "state_dict": best_agent.state_dict(),
                "input_dim": int(np.prod(env.observation_space.shape)),
                "output_dim": env.action_space.n,
                "env_name": env_name,
            },
            ea_checkpoint_path(env_name),
        )
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = make_env(eval_maze, optimal_path=eval_path, env_name=env_name)
        success_rate, avg_ratio = evaluate_policy("ea", best_agent, eval_env, eval_optimal)
        print(f"EA eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")

    elif algo == "qlearn":
        import time
        start_time = time.time()

        agent, rewards, successes, steps_per_episode = train_q_learning(
            env, episodes=episodes, max_steps=env.max_steps
        )
        duration = time.time() - start_time
        print(f"Q-learning training time: {duration:.2f} seconds")

        np.save(os.path.join(MODEL_DIR, "qlearn_rewards.npy"), np.array(rewards, dtype=np.float32))
        suffix = maze_type if maze_type else "simple"
        agent.save(os.path.join(MODEL_DIR, f"qlearn_{suffix}.pkl"))
        # Save Q-table to CSV
        q_csv = os.path.join(MODEL_DIR, f"q_table_{suffix}.csv")
        with open(q_csv, "w") as f:
            for state, values in agent.q_table.items():
                f.write(f"{state}," + ",".join(f"{v:.4f}" for v in values) + "\n")
        print(f"Saved Q-table to {q_csv}")

        step_ratios = [s / max(1, optimal) for s in steps_per_episode]
        success_curve = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        step_curve = np.cumsum(step_ratios) / np.arange(1, len(step_ratios) + 1)
        plot_training_curves(
            os.path.join(MODEL_DIR, "qlearn_training.png"),
            rewards=rewards,
            success_curve=success_curve,
            step_ratio_curve=step_curve,
            title="Q-Learning Training"
        )
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = make_env(eval_maze, optimal_path=eval_path, env_name=env_name)
        success_rate, avg_ratio = evaluate_policy("qlearn", agent, eval_env, eval_optimal)
        print(f"Q-learning eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")
        
    elif algo == "imitation":
        print("Building expert trajectories for imitation...")
        obs_data, act_data = build_imitation_dataset(limit=300, split="train", env_name=env_name)
        if obs_data is None:
            print("Imitation training aborted: no expert data.")
            return
        obs_dim = int(np.prod(env.observation_space.shape))
        agent = ImitationAgent(obs_dim, env.action_space.n)
        batch_size = 64
        epochs = max(5, episodes // 10)
        epoch_losses = []
        for epoch in range(epochs):
            perm = np.random.permutation(len(obs_data))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(obs_data), batch_size):
                idx = perm[start:start+batch_size]
                loss = agent.train_step(obs_data[idx], act_data[idx])
                epoch_loss += loss
                batches += 1
            avg_loss = epoch_loss / max(1, batches)
            epoch_losses.append(avg_loss)
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")
        torch.save(agent.state_dict(), f"{MODEL_DIR}/imitation_{env_name}.pth")
        plot_training_curves(
            os.path.join(MODEL_DIR, "imitation_training.png"),
            losses=epoch_losses,
            title="Imitation Training Loss"
        )
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = make_env(eval_maze, optimal_path=eval_path, env_name=env_name)
        success_rate, avg_ratio = evaluate_policy("imitation", agent, eval_env, eval_optimal)
        print(f"Imitation eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")

    print(f"{algo} training complete.")

def visualize(algo, difficulty, env_name="simple", maze_type=None):
    maze_bundle = load_random_maze(difficulty, split="test", maze_type=maze_type)
    if maze_bundle is None:
        print("No maze available to visualize. Generate data first.")
        return
    maze, optimal_steps, path = maze_bundle
    env = make_env(maze, optimal_path=path, render_mode="human", env_name=env_name)
    
    model = None
    
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(f"{MODEL_DIR}/ppo_mouse")
        elif algo == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(f"{MODEL_DIR}/dqn_mouse")
        elif algo == "ea":
            from src.agents import SimpleEvolutionaryAgent
            ckpt_path = ea_checkpoint_path(env_name)
            # Allow legacy checkpoints: try safe weights-only load, then fallback.
            import torch.serialization as ts

            ts.add_safe_globals([np.core.multiarray.scalar])
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                input_dim = ckpt.get("input_dim", int(np.prod(env.observation_space.shape)))
                output_dim = ckpt.get("output_dim", env.action_space.n)
            else:
                state_dict = ckpt
                input_dim = int(np.prod(env.observation_space.shape))
                output_dim = env.action_space.n

            expected_in = int(np.prod(env.observation_space.shape))
            expected_out = env.action_space.n
            if input_dim != expected_in or output_dim != expected_out:
                print(f"EA checkpoint mismatch (ckpt dims {input_dim}/{output_dim} vs env {expected_in}/{expected_out}). Train EA for env '{env_name}' or load matching checkpoint.")
                return

            model = SimpleEvolutionaryAgent(input_dim, output_dim)
            missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
            if missing_keys:
                print("EA checkpoint architecture mismatch (missing keys). Retrain EA for this env after the LSTM change.")
                return
            model.load_state_dict(state_dict, strict=False)
        elif algo == "qlearn":
            model = QLearningAgent(action_size=env.action_space.n)
            suffix = args.complexity if args.complexity else "simple"
            model.load(os.path.join(MODEL_DIR, f"qlearn_{suffix}.pkl"))
        elif algo == "imitation":
            obs_dim = int(np.prod(env.observation_space.shape))
            model = ImitationAgent(obs_dim, env.action_space.n)
            model.load_state_dict(torch.load(f"{MODEL_DIR}/imitation_{env_name}.pth"))
    except FileNotFoundError:
        print(f"Model {algo} not found. Train it first.")
        return

    print(f"Visualizing {algo} on {difficulty} maze. Optimal Steps: {optimal_steps}")
    
    obs, _ = env.reset()
    if algo == "ea" and hasattr(model, "reset_state"):
        model.reset_state()
    if algo == "imitation" and hasattr(model, "reset_state"):
        model.reset_state()
    done = False
    
    while not done:
        if algo in ["ppo", "dqn"]:
            action, _ = model.predict(obs, deterministic=True)
        elif algo == "ea":
            action = model.select_action(obs)
        elif algo == "qlearn":
            action = model.act(obs, explore=False)
        elif algo == "imitation":
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Check events
        if terminated:
            print("Mouse found the cheese!")
    
    # Keep the window open until user acknowledges (prevents instant close)
    if env.render_mode == "human":
        try:
            input("Press Enter to close the visualization window...")
        except EOFError:
            pass

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train", "visualize", "evaluate"], required=True)
    parser.add_argument("--algo", choices=["ppo", "dqn", "ea", "imitation", "qlearn"], default="ppo")
    parser.add_argument("--difficulty", choices=["Easy", "Medium", "Hard"], default="Medium")
    parser.add_argument("--complexity", choices=["perfect", "imperfect"], default=None, help="Maze type filter (used for qlearn/visualize).")
    parser.add_argument(
        "--env",
        choices=["simple", "mouse"],
        default="simple",
        help="Environment type to use.",
    )
    parser.add_argument("--eval-runs", type=int, default=50, help="Number of evaluation runs (for evaluate mode).")
    
    args = parser.parse_args()
    env_name = args.env
    
    if args.mode == "generate":
        gen = MazeGenerator(DATA_DIR_BASE)
        gen.generate_dataset()
    elif args.mode == "train":
        # Use more episodes for Q-learning on Medium mazes
        eps = 5000 if args.algo == "qlearn" else 1000
        train(args.algo, episodes=eps, env_name=env_name, difficulty=args.difficulty, maze_type=args.complexity)
    elif args.mode == "visualize":
        visualize(args.algo, args.difficulty, env_name=env_name, maze_type=args.complexity)
    elif args.mode == "evaluate":
        if args.algo != "imitation":
            print("Evaluate mode currently supports imitation only.")
            sys.exit(1)
        evaluate_imitation(args.difficulty, env_name=env_name, runs=args.eval_runs)
