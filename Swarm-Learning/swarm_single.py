import argparse
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow imports from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.maze_generator import CELL_WALL, CELL_START, CELL_GOAL  # noqa: E402
from swarm_ea import SwarmAgent, EAConfig, mutate, crossover  # noqa: E402

# Paths/config
DATA_DIR = os.path.join(ROOT, "src", "data", "3131")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
MODEL_PATH = os.path.join(ROOT, "trained_models", "swarm_single_best.pth")
PLOT_PATH = os.path.join(METRICS_DIR, "swarm_single_path.png")

# Defaults for single-maze training
DEFAULT_FILENAME = "maze_2301.npy"  # shortest path_len imperfect maze
ROLLOUTS_PER_AGENT = 3


def load_maze(filename: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Maze file not found: {path}")
    return np.load(path)


def evaluate_agent_single(
    agent: SwarmAgent,
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    cfg: EAConfig,
) -> float:
    scores: List[float] = []
    for _ in range(ROLLOUTS_PER_AGENT):
        scores.append(
            agent.fitness(
                grid,
                start_pos,
                goal_pos,
                epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end,
                action_noise_std=cfg.action_noise_std,
            )
        )
    return float(np.mean(scores))


def train_single(grid: np.ndarray, cfg: EAConfig):
    start_pos = tuple(np.argwhere(grid == CELL_START)[0])
    goal_pos = tuple(np.argwhere(grid == CELL_GOAL)[0])

    population = [SwarmAgent(hidden_size=cfg.hidden_size) for _ in range(cfg.pop_size)]
    best_agent = None
    best_score = -np.inf

    for gen in range(cfg.generations):
        scores = [evaluate_agent_single(agent, grid, start_pos, goal_pos, cfg) for agent in population]
        ranked = [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]
        if scores[np.argmax(scores)] > best_score:
            best_score = scores[np.argmax(scores)]
            best_agent = ranked[0]

        elite_count = max(1, int(cfg.elite_frac * cfg.pop_size))
        elites = ranked[:elite_count]
        new_pop = elites.copy()
        while len(new_pop) < cfg.pop_size:
            parent_a, parent_b = (
                np.random.choice(elites, size=2, replace=True) if len(elites) > 1 else (elites[0], elites[0])
            )
            if np.random.rand() < cfg.crossover_prob:
                base_child = crossover(parent_a, parent_b)
            else:
                base_child = parent_a
            child = mutate(base_child, cfg)
            new_pop.append(child)
        population = new_pop

        if (gen + 1) % 5 == 0 or gen == 0:
            print(
                "[EA Single] Gen {}/{} best={:.2f} mean={:.2f} median={:.2f} p90={:.2f}".format(
                    gen + 1, cfg.generations, max(scores), np.mean(scores), np.median(scores), np.percentile(scores, 90)
                )
            )

    return best_agent, best_score


def visualize(best_agent: SwarmAgent, grid: np.ndarray, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
    path = best_agent.simulate(grid, start_pos, goal_pos)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow((grid == CELL_WALL).astype(int), cmap="gray_r", origin="upper")
    sy, sx = start_pos
    gy, gx = goal_pos
    ax.scatter(sx, sy, c="blue", label="Start")
    ax.scatter(gx, gy, c="red", label="Goal")
    ys, xs = zip(*path)
    ax.plot(xs, ys, c="green", alpha=0.7, label="Path")
    ax.set_title("Best Swarm Agent Path (single maze)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"[EA Single] Saved path plot to {PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train swarm EA on a single 31x31 imperfect maze.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Maze .npy filename in data/3131")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-std", type=float, default=0.15)
    parser.add_argument("--crossover-prob", type=float, default=0.7)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--eps-start", type=float, default=0.25)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--action-noise-std", type=float, default=0.05)
    parser.add_argument("--weight-clip", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    grid = load_maze(args.filename)
    start_pos = tuple(np.argwhere(grid == CELL_START)[0])
    goal_pos = tuple(np.argwhere(grid == CELL_GOAL)[0])
    print(f"[EA Single] Training on {args.filename} start={start_pos} goal={goal_pos}")

    cfg = EAConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_std=args.mutation_std,
        crossover_prob=args.crossover_prob,
        hidden_size=args.hidden_size,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        action_noise_std=args.action_noise_std,
        weight_clip=args.weight_clip,
    )

    best_agent, best_score = train_single(grid, cfg)
    torch.save({"W1": best_agent.W1, "W2": best_agent.W2}, MODEL_PATH)
    print(f"[EA Single] Done. Best score={best_score:.2f}. Saved weights to {MODEL_PATH}")

    visualize(best_agent, grid, start_pos, goal_pos)


if __name__ == "__main__":
    main()
