import os
import random
import time
from collections import deque

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Use the legacy mouse environment (4 actions, FoV obs)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys  # noqa: E402

sys.path.append(ROOT)
from src.mouse_env import MouseMazeEnvLegacy  # noqa: E402

# ---------------------------------
# Hyperparameters / paths
# ---------------------------------
DATA_DIR = os.path.join(ROOT, "src", "data", "2121")
META_PATH = os.path.join(DATA_DIR, "maze_metadata.csv")
MODEL_PATH = os.path.join(ROOT, "trained_models", "dqn_model_multi_improved.pth")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
REWARD_PLOT = os.path.join(METRICS_DIR, "dqn_multi_improved_rewards.png")
SUCCESS_PLOT = os.path.join(METRICS_DIR, "dqn_multi_improved_success.png")
TEST_METRICS_CSV = os.path.join(METRICS_DIR, "dqn_multi_improved_test_metrics.csv")
GIF_PATH = os.path.join(METRICS_DIR, "dqn_multi_improved_eval.gif")

EPISODES = 15000
MAX_STEPS = 4000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999
LR = 0.0001
BATCH_SIZE = 64
MEMORY_SIZE = 80_000
TARGET_UPDATE = 1000
HIDDEN_SIZE = 128
SEQ_LEN = 12
BURN_IN = 4
STALL_LIMIT = 50  # break episodes early when no new cells visited
USE_CURRICULUM = True  # reverse curriculum over path length


# ---------------------------------
# Model: simple recurrent DQN (zeroed hidden for replay batches)
# ---------------------------------
class RecurrentDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=HIDDEN_SIZE, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, x):
        # Stateless forward (used for greedy action selection)
        logits, _ = self.forward_step(x, None)
        return logits

    def forward_step(self, obs_t: torch.Tensor, hidden=None):
        """
        obs_t: (batch, obs_dim)
        hidden: (h, c) or None
        returns logits (batch, actions), new_hidden
        """
        x = self.embed(obs_t)
        x = x.unsqueeze(1)  # (batch, 1, hidden)
        out, hidden = self.rnn(x, hidden)
        logits = self.head(out[:, -1, :])
        return logits, hidden


# ---------------------------------
# Replay buffer
# ---------------------------------
class EpisodeBuffer:
    def __init__(self, capacity_steps, burn_in=BURN_IN, seq_len=SEQ_LEN):
        self.capacity = capacity_steps
        self.episodes = deque()
        self.current = []
        self.total_steps = 0
        self.burn_in = burn_in
        self.seq_len = seq_len

    def push(self, s, a, r, ns, done):
        self.current.append((s, a, r, ns, done))
        self.total_steps += 1
        if done:
            self._finalize_episode()
        self._trim()

    def _finalize_episode(self):
        if self.current:
            self.episodes.append(self.current)
        self.current = []

    def _trim(self):
        # ensure total steps across episodes <= capacity
        while self.total_steps > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.total_steps -= len(removed)

    def sample_segment(self):
        # pick an episode with enough length
        eligible = [ep for ep in self.episodes if len(ep) > self.burn_in + self.seq_len]
        if not eligible:
            return None
        ep = random.choice(eligible)
        start = random.randint(self.burn_in, len(ep) - self.seq_len)
        context_start = start - self.burn_in
        context = ep[context_start:start]
        segment = ep[start:start + self.seq_len]
        return context, segment

    def __len__(self):
        return self.total_steps


# ---------------------------------
# Utilities
# ---------------------------------
def load_metadata():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")
    meta = np.genfromtxt(META_PATH, delimiter=",", names=True, dtype=None, encoding=None)
    records = [dict(zip(meta.dtype.names, row)) for row in meta]
    if "split" not in meta.dtype.names:
        random.shuffle(records)
        split_idx = int(len(records) * 0.9)
        for i, rec in enumerate(records):
            rec["split"] = "train" if i < split_idx else "test"

    # Compute path_len if missing (prefer optimal_steps)
    for rec in records:
        if "path_len" in rec:
            continue
        if "optimal_steps" in rec and rec["optimal_steps"] not in (None, ""):
            rec["path_len"] = rec["optimal_steps"]
        else:
            rec["path_len"] = 0
    return records


def curriculum_sampler(train_recs, current_ep, total_eps):
    if not USE_CURRICULUM:
        return random.choice(train_recs)
    difficulty_p = current_ep / max(1, total_eps)
    sorted_recs = sorted(train_recs, key=lambda x: x.get("path_len", 0))
    num_mazes = len(sorted_recs)
    end_idx = int(num_mazes * (0.2 + 0.8 * difficulty_p))
    end_idx = max(1, min(end_idx, num_mazes))
    return random.choice(sorted_recs[:end_idx])


def make_env(filename):
    maze = np.load(os.path.join(DATA_DIR, filename))
    if 2 not in maze or 3 not in maze:
        raise ValueError(f"Maze {filename} missing start=2 or goal=3 marker.")
    # Reduce shaping by disabling smell/optimal penalties
    env = MouseMazeEnvLegacy(
        maze,
        optimal_path=None,
        render_mode=None,
        use_smell=False,
        smell_in_obs=False,
        use_optimal_penalty=False,
    )
    return env


# ---------------------------------
# Training loop
# ---------------------------------
def train():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    records = load_metadata()
    train_recs = [r for r in records if r["split"] == "train"]
    test_recs = [r for r in records if r["split"] == "test"]

    # Probe env for dimensions
    probe_env = make_env(train_recs[0]["filename"])
    obs_dim = probe_env.observation_space.shape[0]
    n_actions = probe_env.action_space.n
    probe_env.close()

    policy = RecurrentDQN(obs_dim, n_actions)
    target = RecurrentDQN(obs_dim, n_actions)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = EpisodeBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    rewards_hist = []
    success_hist = []
    total_steps = 0

    for ep in range(EPISODES):
        rec = curriculum_sampler(train_recs, ep, EPISODES)
        env = make_env(rec["filename"])
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        success = 0
        steps = 0
        visited = set()
        last_new_step = 0

        for _ in range(min(MAX_STEPS, env.max_steps)):
            steps += 1
            total_steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    q = policy(obs_t).squeeze(0)
                    action = int(torch.argmax(q).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            pos = tuple(env.agent_pos)
            if pos not in visited:
                visited.add(pos)
                last_new_step = steps
            elif steps - last_new_step > STALL_LIMIT:
                reward -= 2.0  # small extra penalty for stalling
                done = True

            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            if terminated:
                success = 1
                break
        env.close()

        # Train
        # Train on sampled segments (with burn-in for hidden state)
        if len(buffer) >= BATCH_SIZE:
            losses = []
            for _ in range(BATCH_SIZE):
                sample = buffer.sample_segment()
                if sample is None:
                    break
                context, segment = sample

                # Build burn-in hidden states
                h_online = None
                h_target = None
                for (s_bi, _, _, _, _) in context:
                    s_bi_t = torch.tensor(s_bi, dtype=torch.float32).unsqueeze(0)
                    _, h_online = policy.forward_step(s_bi_t, h_online)
                    with torch.no_grad():
                        _, h_target = target.forward_step(s_bi_t, h_target)

                # Compute loss over segment
                seg_loss = 0.0
                for (s_tr, a_tr, r_tr, ns_tr, d_tr) in segment:
                    s_t = torch.tensor(s_tr, dtype=torch.float32).unsqueeze(0)
                    ns_t = torch.tensor(ns_tr, dtype=torch.float32).unsqueeze(0)
                    logits, h_online = policy.forward_step(s_t, h_online)
                    q_sa = logits.gather(1, torch.tensor([[a_tr]]))
                    with torch.no_grad():
                        next_logits, h_target = target.forward_step(ns_t, h_target)
                        next_q = next_logits.max(dim=1, keepdim=True)[0]
                        target_q = torch.tensor([[r_tr]], dtype=torch.float32) + GAMMA * next_q * (1 - d_tr)
                    seg_loss = seg_loss + nn.MSELoss()(q_sa, target_q)

                if isinstance(seg_loss, torch.Tensor):
                    losses.append(seg_loss)

            if losses:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

        # Update target
        if ep % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

        # Epsilon decay
        epsilon = max(EPSILON_MIN, epsilon * 0.99995)

        rewards_hist.append(ep_reward)
        success_hist.append(success)

        if ep % 50 == 0:
            elapsed = (time.time() - start_time) / 60.0 if ep > 0 else 0.0
            sr = np.mean(success_hist[-100:]) if len(success_hist) >= 10 else np.mean(success_hist)
            print(f"Ep {ep}/{EPISODES} | ε={epsilon:.3f} | success(100)={sr:.2f} | steps={total_steps} | elapsed={elapsed:.1f}m")

    # Save model
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"Saved improved DQN model to {MODEL_PATH}")

    # Reward curve
    plt.figure()
    plt.plot(rewards_hist)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Multi Improved - Training Reward")
    plt.grid(True)
    plt.savefig(REWARD_PLOT)
    plt.close()
    print(f"Saved reward plot to {REWARD_PLOT}")

    # Success curve (moving average)
    window = 200
    if len(success_hist) >= window:
        ma = np.convolve(success_hist, np.ones(window) / window, mode="valid")
    else:
        ma = np.array(success_hist)
    plt.figure()
    plt.plot(ma)
    plt.xlabel("Episode")
    plt.ylabel(f"Success (window={min(window,len(success_hist))})")
    plt.title("DQN Multi Improved - Success Rate")
    plt.grid(True)
    plt.savefig(SUCCESS_PLOT)
    plt.close()
    print(f"Saved success plot to {SUCCESS_PLOT}")

    # Evaluate on test set
    test_rows = test_recs if len(test_recs) > 0 else train_recs[:100]
    eval_records = []
    for rec in test_rows:
        env = make_env(rec["filename"])
        obs, _ = env.reset()
        steps = 0
        success = False
        for _ in range(min(MAX_STEPS, env.max_steps)):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q = policy(obs_t)
                action = int(torch.argmax(q).item())
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated:
                success = True
                break
            if truncated:
                break
        env.close()
        eval_records.append(
            {"filename": rec["filename"], "success": success, "steps": steps}
        )

    import csv

    with open(TEST_METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "success", "steps"])
        writer.writeheader()
        writer.writerows(eval_records)
    test_success = np.mean([r["success"] for r in eval_records])
    print(f"Test success rate: {test_success:.2f} ({len(eval_records)} mazes)")
    print(f"Saved test metrics to {TEST_METRICS_CSV}")

    # GIF on first test maze
    if len(test_rows) > 0:
        rec = test_rows[0]
        env = make_env(rec["filename"])
        obs, _ = env.reset()
        frames = []
        max_eval_steps = min(MAX_STEPS, env.max_steps)
        for _ in range(max_eval_steps):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q = policy(obs_t)
                action = int(torch.argmax(q).item())
            obs, _, terminated, truncated, _ = env.step(action)
            try:
                surf = pygame.display.get_surface()
            except Exception:
                surf = None
            if surf is not None:
                frame = pygame.surfarray.array3d(surf).swapaxes(0, 1)
                frames.append(frame)
            if terminated or truncated:
                break
        env.close()
        if frames:
            imageio.mimsave(GIF_PATH, frames, fps=10, format="GIF")
            print(f"Saved eval GIF to {GIF_PATH}")
        else:
            print("No frames captured for GIF (is a display available?).")


if __name__ == "__main__":
    start_time = time.time()
    train()
