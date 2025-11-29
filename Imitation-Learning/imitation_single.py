import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pygame
from collections import deque

# Allow imports from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.mouse_env import MouseMazeEnvLegacy  # noqa: E402
from src.agents import ImitationAgent  # noqa: E402


def direction_to_action(prev, curr):
    dr = curr[0] - prev[0]
    dc = curr[1] - prev[1]
    mapping = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    return mapping.get((dr, dc), None)


def collect_path_data(maze, path):
    env = MouseMazeEnvLegacy(maze, optimal_path=path, render_mode=None)
    obs_list, act_list = [], []
    obs, _ = env.reset()
    path_seq = path.tolist()
    for i in range(1, len(path_seq)):
        prev = tuple(path_seq[i - 1])
        curr = tuple(path_seq[i])
        action = direction_to_action(prev, curr)
        if action is None:
            continue
        obs_list.append(obs)
        obs, _, _, _, _ = env.step(action)
        act_list.append(action)
    env.close()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def train_single(maze_path, metrics_dir, model_path, epochs=450, batch_size=64):
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    maze = np.load(maze_path)
    # Load expert path if present
    path_path = maze_path.replace(".npy", "_path.npy")
    if not os.path.exists(path_path):
        raise FileNotFoundError(f"Path file not found: {path_path}")
    path = np.load(path_path)

    obs_data, act_data = collect_path_data(maze, path)
    obs_dim = obs_data.shape[1]
    n_actions = 4

    agent = ImitationAgent(obs_dim, n_actions)

    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(len(obs_data))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(obs_data), batch_size):
            idx = perm[start:start + batch_size]
            loss = agent.train_step(obs_data[idx], act_data[idx])
            epoch_loss += loss
            batches += 1
        avg_loss = epoch_loss / max(1, batches)
        losses.append(avg_loss)
        agent.step_scheduler(avg_loss)
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"[Single] Epoch {epoch+1}/{epochs} loss: {avg_loss:.4f}")

    torch.save(agent.state_dict(), model_path)
    print(f"Saved imitation single model to {model_path}")

    # Evaluate on the same maze with visualization (legacy env)
    env = MouseMazeEnvLegacy(maze, optimal_path=path, render_mode="human")
    obs, _ = env.reset()
    frames = []
    terminated = False
    truncated = False
    steps = 0
    agent.reset_state()
    while not (terminated or truncated) and steps < env.max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        surf = None
        try:
            surf = pygame.display.get_surface()
        except Exception:
            surf = None
        if surf is not None:
            frame = pygame.surfarray.array3d(surf).swapaxes(0, 1)
            frames.append(frame)
    env.close()

    success = terminated
    optimal_steps = len(path)
    step_ratio = steps / max(1, optimal_steps)
    print(f"[Single Eval] success={success}, steps={steps}, optimal={optimal_steps}, step_ratio={step_ratio:.2f}")

    if frames:
        gif_path = os.path.join(metrics_dir, "imitation_single_eval.gif")
        imageio.mimsave(gif_path, frames, fps=10, format="GIF")
        print(f"Saved eval GIF to {gif_path}")

    # Loss plot with eval summary in title
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Imitation Single Loss | success={success} steps={steps} optimal={optimal_steps} ratio={step_ratio:.2f}")
    plt.grid(True)
    loss_path = os.path.join(metrics_dir, "imitation_single_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved loss plot to {loss_path}")


if __name__ == "__main__":
    METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
    MODEL_PATH = os.path.join(ROOT, "trained_models", "imitation_single.pth")
    # Default to first hard (31x31) maze path file in src/data/3131
    DEFAULT_MAZE = os.path.join(ROOT, "src", "data", "3131", "maze_0.npy")
    train_single(DEFAULT_MAZE, METRICS_DIR, MODEL_PATH)
