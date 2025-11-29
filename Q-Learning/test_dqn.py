import os
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import pygame
from maze_env import MazeEnv
from DQN_Single import DQN, one_hot_state, MAX_STEPS


def run_single(model_path, maze_path, render=False, gif_path=None):
    maze = np.load(maze_path)
    env = MazeEnv(maze, render_mode="human" if render else None)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    model = DQN(n_states, n_actions)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    state, _ = env.reset()
    state_vec = one_hot_state(state, n_states)

    frames = []
    max_steps = min(MAX_STEPS, env.max_steps)
    for step in range(max_steps):
        with torch.no_grad():
            action = torch.argmax(model(torch.tensor(state_vec))).item()
        state, reward, terminated, truncated, _ = env.step(action)
        state_vec = one_hot_state(state, n_states)

        if render:
            surf = None
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

    if gif_path and frames:
        imageio.mimsave(gif_path, frames, fps=10, format="GIF")
        print(f"Saved run GIF to {gif_path}")
    return terminated, step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../trained_models/dqn_model_curriculum.pth", help="Path to trained DQN model")
    parser.add_argument("--maze", default="../src/data/2121/maze_757.npy", help="Path to maze .npy file")
    parser.add_argument("--render", action="store_true", help="Show pygame window")
    parser.add_argument("--gif", default="metrics/dqn_test.gif", help="Path to save GIF for single run")
    parser.add_argument("--batch", nargs="*", help="Optional list of maze files to batch test")
    args = parser.parse_args()

    os.makedirs("metrics", exist_ok=True)

    if args.batch:
        results = []
        for mpath in args.batch:
            success, steps = run_single(args.model, mpath, render=False, gif_path=None)
            results.append((mpath, success, steps))
            print(f"{mpath}: success={success}, steps={steps}")
        # Summary
        successes = sum(1 for _, s, _ in results if s)
        print(f"Batch complete: {successes}/{len(results)} successes")
    else:
        success, steps = run_single(args.model, args.maze, render=args.render, gif_path=args.gif)
        print(f"Single run: success={success}, steps={steps}")


if __name__ == "__main__":
    main()
