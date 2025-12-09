import os
import sys
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.mouse_env import MouseMazeEnvLegacy  # noqa: E402

# Paths
DATA_DIR = os.path.join(ROOT, "src", "data", "3131")
META_PATH = os.path.join(DATA_DIR, "maze_metadata.csv")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
MODEL_PATH = os.path.join(ROOT, "trained_models", "ppo_bc.pth")
EVAL_CSV = os.path.join(METRICS_DIR, "ppo_bc_eval.csv")

# PPO hyperparameters
ROLLOUT_STEPS = 2048
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 256
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# BC regularization
USE_FLOW_FIELD = True
BC_COEF = 0.1
BC_BATCH = 512

# Curriculum + stall handling
USE_CURRICULUM = True
STALL_LIMIT = 50
STALL_PENALTY = -2.0
TOTAL_UPDATES = 1500  # adjust based on compute budget


def load_metadata():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")
    meta = pd.read_csv(META_PATH)
    if "split" not in meta.columns:
        meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(meta) * 0.9)
        meta["split"] = "train"
        meta.loc[split_idx:, "split"] = "test"
        meta.to_csv(META_PATH, index=False)

    if "path_len" not in meta.columns:
        meta["path_len"] = meta.get("optimal_steps", np.nan)
    meta["path_len"] = meta["path_len"].fillna(meta["path_len"].median())
    return meta


def collect_flow_dataset(df):
    obs_list = []
    act_list = []
    for _, row in df.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        action_file = row.get("action_file", None)
        if not isinstance(action_file, str):
            continue
        action_path = os.path.join(DATA_DIR, action_file)
        if not os.path.exists(action_path):
            continue
        action_map = np.load(action_path)
        if 2 not in maze or 3 not in maze:
            continue
        env = MouseMazeEnvLegacy(maze, optimal_path=None, render_mode=None)
        obs, _ = env.reset()
        steps = 0
        max_steps = env.max_steps
        while steps < max_steps:
            r, c = env.agent_pos
            act = int(action_map[r, c])
            if act < 0:
                break
            obs_list.append(obs)
            obs, _, terminated, truncated, _ = env.step(act)
            act_list.append(act)
            steps += 1
            if terminated or truncated:
                break
        env.close()
    if len(obs_list) == 0:
        return None, None
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class PPONet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.policy_head(x), self.value_head(x)


def make_env(filename):
    maze = np.load(os.path.join(DATA_DIR, filename))
    env = MouseMazeEnvLegacy(
        maze,
        optimal_path=None,
        render_mode=None,
        use_smell=False,
        smell_in_obs=False,
        use_optimal_penalty=False,
    )
    return env


def curriculum_sampler(train_df, current_update):
    if not USE_CURRICULUM:
        return train_df.sample(1).iloc[0]
    difficulty_p = current_update / max(1, TOTAL_UPDATES)
    sorted_df = train_df.sort_values("path_len")
    end_idx = int(len(sorted_df) * (0.2 + 0.8 * difficulty_p))
    end_idx = max(1, min(end_idx, len(sorted_df)))
    return sorted_df.iloc[random.randint(0, end_idx - 1)]


def rollout(env, policy, device):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    visited = set()
    last_new = 0
    hidden = None  # feedforward
    for t in range(ROLLOUT_STEPS):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        pos = tuple(env.agent_pos)
        if pos not in visited:
            visited.add(pos)
            last_new = t
        elif (t - last_new) > STALL_LIMIT:
            reward += STALL_PENALTY
            done = True

        obs_buf.append(obs_t.squeeze(0))
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(torch.tensor([reward], device=device))
        val_buf.append(value.squeeze(0))
        done_buf.append(done)

        obs = next_obs
        if done:
            obs, _ = env.reset()
            visited = set()
            last_new = 0

    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_val = policy(obs_t)
    last_val = last_val.squeeze(0)

    # GAE-Lambda
    advantages = []
    gae = torch.zeros(1, device=device)
    for t in reversed(range(ROLLOUT_STEPS)):
        mask = 0.0 if done_buf[t] else 1.0
        delta = rew_buf[t] + GAMMA * mask * (val_buf[t + 1] if t + 1 < len(val_buf) else last_val) - val_buf[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, val_buf)]

    batch = PPOBatch(
        obs=torch.stack(obs_buf),
        actions=torch.stack(act_buf),
        logprobs=torch.stack(logp_buf),
        returns=torch.stack(returns).detach(),
        advantages=torch.stack(advantages).detach(),
        values=torch.stack(val_buf).detach(),
    )
    return batch


def ppo_update(policy, optimizer, batch: PPOBatch, bc_data, device):
    obs, actions, old_logprobs, returns, advantages = (
        batch.obs,
        batch.actions,
        batch.logprobs.detach(),
        batch.returns,
        batch.advantages,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    idxs = np.arange(len(obs))
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, len(obs), PPO_BATCH_SIZE):
            mb_idx = idxs[start : start + PPO_BATCH_SIZE]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logp = old_logprobs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            logits, values = policy(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - values.squeeze(-1)).pow(2).mean()

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            # BC regularization
            if USE_FLOW_FIELD and bc_data[0] is not None and BC_COEF > 0:
                obs_bc, act_bc = bc_data
                bc_idx = np.random.choice(len(obs_bc), size=min(BC_BATCH, len(obs_bc)), replace=False)
                bc_obs = torch.tensor(obs_bc[bc_idx], dtype=torch.float32, device=device)
                bc_act = torch.tensor(act_bc[bc_idx], dtype=torch.long, device=device)
                bc_logits, _ = policy(bc_obs)
                bc_loss = nn.CrossEntropyLoss()(bc_logits, bc_act)
                loss = loss + BC_COEF * bc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def evaluate(policy, df):
    records = []
    for _, row in df.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        env = MouseMazeEnvLegacy(maze, optimal_path=None, render_mode=None)
        obs, _ = env.reset()
        steps = 0
        success = False
        max_steps = int(env.max_steps * 2)
        with torch.no_grad():
            while steps < max_steps:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = policy(obs_t)
                action = torch.argmax(logits, dim=1).item()
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated:
                    success = True
                    break
                if truncated:
                    break
        env.close()
        optimal_steps = row.get("optimal_steps", None)
        step_ratio = steps / optimal_steps if success and optimal_steps and optimal_steps > 0 else np.nan
        records.append(
            {
                "filename": row["filename"],
                "success": success,
                "steps": steps,
                "step_ratio": step_ratio,
            }
        )
    return pd.DataFrame(records)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    meta = load_metadata()
    train_df = meta[meta["split"] == "train"]
    test_df = meta[meta["split"] == "test"]

    # Flow dataset for BC regularization
    bc_obs, bc_act = collect_flow_dataset(train_df) if USE_FLOW_FIELD else (None, None)
    if bc_obs is None:
        print("Flow-field data missing; BC regularization disabled.")
        bc_data = (None, None)
    else:
        bc_data = (bc_obs, bc_act)
        print(f"Loaded flow-field BC dataset: {len(bc_obs)} samples")

    # Probe env
    probe_env = make_env(train_df.iloc[0]["filename"])
    obs_dim = probe_env.observation_space.shape[0]
    n_actions = probe_env.action_space.n
    probe_env.close()

    policy = PPONet(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    env = None
    for update in range(TOTAL_UPDATES):
        rec = curriculum_sampler(train_df, update)
        if env is not None:
            env.close()
        env = make_env(rec["filename"])

        batch = rollout(env, policy, device)
        ppo_update(policy, optimizer, batch, bc_data, device)

        if (update + 1) % 50 == 0:
            torch.save(policy.state_dict(), MODEL_PATH)
            print(f"[PPO] Saved checkpoint at update {update+1}")

        if (update + 1) % 200 == 0:
            policy.eval()
            eval_df = evaluate(policy, test_df.sample(min(200, len(test_df))))
            success = eval_df["success"].mean()
            print(f"[PPO] Update {update+1}: test success={success:.3f}")
            policy.train()

    torch.save(policy.state_dict(), MODEL_PATH)
    policy.eval()
    train_eval = evaluate(policy, train_df)
    test_eval = evaluate(policy, test_df)
    train_eval.to_csv(os.path.join(METRICS_DIR, "ppo_bc_train_eval.csv"), index=False)
    test_eval.to_csv(os.path.join(METRICS_DIR, "ppo_bc_test_eval.csv"), index=False)
    print(f"[PPO] Final train_success={train_eval['success'].mean():.3f}, test_success={test_eval['success'].mean():.3f}")


if __name__ == "__main__":
    main()
