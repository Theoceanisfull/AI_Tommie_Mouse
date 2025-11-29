import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pygame
from maze_env import MazeEnv

EPISODES = 25000
MAX_STEPS = 5000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.25
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 80000
TARGET_UPDATE = 200
# Medium (21x21) dataset location; adjust if you generated under src/data
DATA_DIR = "../src/data/2121" if os.path.exists("../src/data/2121") else "../data/2121"
MODEL_PATH = "../trained_models/dqn_model_multi.pth"
METRICS_DIR = "metrics"
REWARD_PLOT = os.path.join(METRICS_DIR, "dqn_multi_rewards.png")
SUCCESS_PLOT = os.path.join(METRICS_DIR, "dqn_multi_success.png")
TEST_METRICS_CSV = os.path.join(METRICS_DIR, "dqn_multi_test_metrics.csv")
GIF_PATH = os.path.join(METRICS_DIR, "dqn_multi_eval.gif")


def get_distance_map(env):
    rows, cols = env.maze.shape
    goal_pos = env.goal_pos
    dist_map = np.full((rows, cols), -1)
    q = deque([(goal_pos, 0)])
    dist_map[goal_pos] = 0
    by_dist = {}
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while q:
        (r, c), d = q.popleft()
        by_dist.setdefault(d, []).append((r, c))
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and env.maze[nr, nc] == 0 and dist_map[nr, nc] == -1:
                dist_map[nr, nc] = d + 1
                q.append(((nr, nc), d + 1))
    return by_dist


def reset_curriculum(env, difficulty_radius, starts_by_dist):
    state, info = env.reset()
    actual_dist = np.random.randint(1, difficulty_radius + 1)
    max_avail_dist = max(starts_by_dist.keys())
    actual_dist = min(actual_dist, max_avail_dist)
    options = starts_by_dist.get(actual_dist, [])
    start_pos = random.choice(options) if options else (0, 0)
    env.agent_pos = start_pos
    env.current_steps = 0
    state_idx = start_pos[0] * env.n_cols + start_pos[1]
    return state_idx, {"position": start_pos}


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def one_hot_state(state, n_states):
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


def load_metadata():
    meta_path = os.path.join(DATA_DIR, "maze_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Generate mazes first.")
    meta = pd.read_csv(meta_path)
    if "split" not in meta.columns:
        meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(meta) * 0.9)
        meta["split"] = "train"
        meta.loc[split_idx:, "split"] = "test"
        meta.to_csv(meta_path, index=False)
    train_meta = meta[meta["split"] == "train"]
    test_meta = meta[meta["split"] == "test"]
    return meta, train_meta, test_meta


def eval_on_set(model, meta_subset, n_states, n_actions, max_steps=MAX_STEPS):
    results = []
    for _, row in meta_subset.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        env = MazeEnv(maze, render_mode=None)
        state, _ = env.reset()
        state_vec = one_hot_state(state, n_states)
        success = False
        steps_taken = 0
        for _ in range(min(max_steps, env.max_steps)):
            with torch.no_grad():
                action = torch.argmax(model(torch.tensor(state_vec))).item()
            state, reward, terminated, truncated, _ = env.step(action)
            state_vec = one_hot_state(state, n_states)
            steps_taken += 1
            if terminated:
                success = True
                break
            if truncated:
                break
        env.close()
        results.append((row["filename"], success, steps_taken))
    return results


def save_eval_gif(model, meta_subset, n_states, n_actions, gif_path=GIF_PATH):
    if meta_subset.empty:
        return
    row = meta_subset.sample(1).iloc[0]
    maze = np.load(os.path.join(DATA_DIR, row["filename"]))
    env = MazeEnv(maze, render_mode="human")
    state, _ = env.reset()
    state_vec = one_hot_state(state, n_states)
    env.render()
    frames = []
    success = False
    steps_taken = 0
    for _ in range(min(MAX_STEPS, env.max_steps)):
        with torch.no_grad():
            action = torch.argmax(model(torch.tensor(state_vec))).item()
        state, reward, terminated, truncated, _ = env.step(action)
        state_vec = one_hot_state(state, n_states)
        steps_taken += 1

        surf = None
        try:
            surf = pygame.display.get_surface()
        except Exception:
            surf = None
        if surf is not None:
            frame = pygame.surfarray.array3d(surf).swapaxes(0, 1)
            frames.append(frame)

        if terminated:
            success = True
            break
        if truncated:
            break
    env.close()
    if frames:
        imageio.mimsave(gif_path, frames, fps=10, format="GIF")
        print(f"Saved eval GIF to {gif_path} (success={success}, steps={steps_taken})")
    else:
        print("No frames captured during eval (headless?).")


if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    meta, train_meta, test_meta = load_metadata()

    # Define spaces from a sample train maze
    sample_maze = np.load(os.path.join(DATA_DIR, train_meta.iloc[0]["filename"]))
    sample_env = MazeEnv(sample_maze, render_mode=None)
    n_states = sample_env.observation_space.n
    n_actions = sample_env.action_space.n

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)

    current_difficulty = 2
    success_buffer = deque(maxlen=20)
    episode_rewards = []
    episode_success = []
    start_time = time.time()
    steps = 0

    base_max_dist = max(get_distance_map(sample_env).keys())
    low_success_streak = 0

    for episode in range(EPISODES):
        # Sample a maze from train set
        maze_row = train_meta.sample(1).iloc[0]
        maze = np.load(os.path.join(DATA_DIR, maze_row["filename"]))
        env = MazeEnv(maze, render_mode=None)

        starts_by_dist = get_distance_map(env)
        maze_max_dist = max(starts_by_dist.keys())

        state, _ = reset_curriculum(env, min(current_difficulty, maze_max_dist), starts_by_dist)
        state_vec = one_hot_state(state, n_states)

        total_reward = 0
        success = 0
        terminated = False
        truncated = False

        current_max_steps = min(MAX_STEPS, min(current_difficulty, maze_max_dist) * 5 + 50)

        for _ in range(current_max_steps):
            steps += 1
            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(torch.tensor(state_vec))).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_vec = one_hot_state(next_state, n_states)
            done = terminated or truncated

            memory.append((state_vec, action, reward, next_state_vec, done))
            state_vec = next_state_vec
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

                states_t = torch.tensor(np.stack(states_b), dtype=torch.float32)
                actions_t = torch.tensor(actions_b, dtype=torch.int64).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1)
                next_states_t = torch.tensor(np.stack(next_states_b), dtype=torch.float32)
                dones_t = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1)

                q_vals = policy_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                    q_target = rewards_t + GAMMA * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_vals, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        if terminated:
            success = 1
        success_buffer.append(success)
        episode_success.append(success)
        success_rate = sum(success_buffer) / len(success_buffer)

        # Curriculum pacing: promote when recent success >0.6 over 20
        if len(success_buffer) >= 20 and success_rate > 0.6:
            if current_difficulty < base_max_dist:
                current_difficulty += 1
                success_buffer.clear()
                print(f"*** Level Up! Increasing Difficulty to Radius {current_difficulty} ***")

        # Epsilon scheduling: warm-up then decay; if success < 0.60 for 10 checks and epsilon is at min, reset to 1.0
        WARMUP_EPISODES = 500
        if episode < WARMUP_EPISODES:
            EPSILON = EPSILON  # keep at current (start is 1.0)
        else:
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            if len(success_buffer) >= 10 and success_rate < 0.60:
                low_success_streak += 1
            else:
                low_success_streak = 0
            if low_success_streak >= 10 and EPSILON <= EPSILON_MIN + 1e-6:
                EPSILON = 1.0
                low_success_streak = 0
                print("*** Low success detected at min epsilon: resetting to 1.0 to re-explore ***")
        episode_rewards.append(total_reward)

        if episode % 20 == 0:
            elapsed = (time.time() - start_time) / 60.0
            mem_len = len(memory)
            print(f"Ep {episode} | Diff: {current_difficulty}/{base_max_dist} | Success: {success_rate:.2f} | Eps: {EPSILON:.3f} | Steps: {steps} | Replay: {mem_len} | Elapsed: {elapsed:.1f} min")

    # Save model
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"Saved multi-maze DQN model to {MODEL_PATH}")

    # Plots
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Multi-Maze Training Rewards")
    plt.grid(True)
    plt.savefig(REWARD_PLOT)
    plt.close()
    print(f"Saved reward plot to {REWARD_PLOT}")

    plt.figure()
    window = 100
    if len(episode_success) >= window:
        rolling = np.convolve(episode_success, np.ones(window) / window, mode="valid")
        plt.plot(rolling)
        plt.xlabel(f"Episode (window={window})")
        plt.ylabel("Success Rate")
        plt.title("Rolling Success Rate")
        plt.grid(True)
        plt.savefig(SUCCESS_PLOT)
        plt.close()
        print(f"Saved success plot to {SUCCESS_PLOT}")

    # Evaluate on test set
    test_results = eval_on_set(policy_net, test_meta, n_states, n_actions)
    test_df = pd.DataFrame(test_results, columns=["filename", "success", "steps"])
    test_df.to_csv(TEST_METRICS_CSV, index=False)
    test_success = test_df["success"].mean()
    test_steps_mean = test_df["steps"].mean()
    print(f"Test success rate: {test_success:.3f}, avg steps: {test_steps_mean:.1f}")
    print(f"Saved test metrics to {TEST_METRICS_CSV}")

    # Save eval GIF on a random test maze
    save_eval_gif(policy_net, test_meta, n_states, n_actions, gif_path=GIF_PATH)

    print(f"Training complete. Final difficulty: {current_difficulty}/{base_max_dist}")
