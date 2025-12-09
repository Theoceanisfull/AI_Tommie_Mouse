import os
import sys
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Allow imports from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.maze_generator import CELL_WALL, CELL_START, CELL_GOAL  # noqa: E402

# Paths/config
DATA_DIR = os.path.join(ROOT, "src", "data", "3131")
META_PATH = os.path.join(DATA_DIR, "maze_metadata.csv")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
MODEL_PATH = os.path.join(ROOT, "trained_models", "swarm_ea_best.pth")
PLOT_PATH = os.path.join(METRICS_DIR, "swarm_best_path.png")

# Evolution hyperparameters (aligned with notebook spirit)
POP_SIZE = 200
GENERATIONS = 200
ELITE_FRAC = 0.2
MUTATION_RATE = 0.1
MUTATION_STD = 0.15
MAZES_PER_GEN = 10
HIDDEN_SIZE = 16
EPS_START = 0.5
EPS_END = 0.02
CURRICULUM = True  # sample easier mazes first (by path_len)
ACTION_NOISE_STD = 0.05
WEIGHT_CLIP = 3.0
WARMUP_EASY_FRAC = 0.4
WARMUP_GENS = 30

# Evaluation settings
EVAL_MAX = 200  # max mazes to evaluate on train/test


@dataclass
class EAConfig:
    pop_size: int = POP_SIZE
    generations: int = GENERATIONS
    elite_frac: float = ELITE_FRAC
    mutation_rate: float = MUTATION_RATE
    mutation_std: float = MUTATION_STD
    crossover_prob: float = 0.7
    mazes_per_gen: int = MAZES_PER_GEN
    hidden_size: int = HIDDEN_SIZE
    epsilon_start: float = EPS_START
    epsilon_end: float = EPS_END
    action_noise_std: float = ACTION_NOISE_STD
    weight_clip: float = WEIGHT_CLIP


class SwarmAgent:
    """
    Notebook-inspired tiny NN: input 7 (goal delta + 4 walls + visited flag), hidden 16, output 4.
    """

    def __init__(self, input_size: int = 7, hidden_size: int = HIDDEN_SIZE, output_size: int = 4, weights=None):
        if weights is None:
            self.W1 = np.random.randn(input_size, hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size)
        else:
            self.W1, self.W2 = weights
        self.reward = 0.0

    def forward(self, x: np.ndarray, noise_std: float = 0.0) -> int:
        h = np.tanh(np.dot(x, self.W1))
        out = np.dot(h, self.W2)
        if noise_std > 0:
            out = out + np.random.randn(*out.shape) * noise_std
        return int(np.argmax(out))

    def simulate(
        self,
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        epsilon_start: float = EPS_START,
        epsilon_end: float = EPS_END,
        action_noise_std: float = ACTION_NOISE_STD,
    ):
        rows, cols = grid.shape
        y, x = start_pos
        path = [(y, x)]
        visited = set()
        visited.add((y, x))
        recent_moves: List[Tuple[int, int]] = []
        self.reward = 0.0
        no_prog = 0
        gy, gx = goal_pos
        max_steps = rows * cols // 2

        for step_idx in range(max_steps):
            # Linearly anneal exploration over the episode
            eps = epsilon_start - (epsilon_start - epsilon_end) * (step_idx / max_steps)
            visited_flag = 1 if (y, x) in visited else 0
            inputs = np.array(
                [
                    (gy - y) / rows,
                    (gx - x) / cols,
                    grid[y - 1, x] if y > 0 else 1,
                    grid[y, x - 1] if x > 0 else 1,
                    grid[y + 1, x] if y < rows - 1 else 1,
                    grid[y, x + 1] if x < cols - 1 else 1,
                    visited_flag,
                ],
                dtype=np.float32,
            )
            move = self.forward(inputs, noise_std=action_noise_std)
            if random.random() < eps:
                move = random.randint(0, 3)

            ny, nx = y, x
            if move == 0:
                ny -= 1
            elif move == 1:
                ny += 1
            elif move == 2:
                nx -= 1
            elif move == 3:
                nx += 1

            self.reward -= 0.02  # step cost

            if not (0 <= ny < rows and 0 <= nx < cols) or grid[ny, nx] == CELL_WALL:
                self.reward -= 1.5
                continue

            prev_y, prev_x = y, x
            prev_dist = np.linalg.norm([prev_y - gy, prev_x - gx])
            y, x = ny, nx
            path.append((y, x))

            # Heading bonus: reward alignment of movement with goal vector
            goal_vec = np.array([gy - prev_y, gx - prev_x], dtype=np.float32)
            move_vec = np.array([y - prev_y, x - prev_x], dtype=np.float32)
            gnorm = np.linalg.norm(goal_vec) + 1e-6
            mnorm = np.linalg.norm(move_vec) + 1e-6
            heading = float(np.dot(goal_vec / gnorm, move_vec / mnorm))
            self.reward += heading * 0.5

            if (y, x) in visited:
                self.reward -= 1.0  # revisit penalty
            visited.add((y, x))

            recent_moves.append((y, x))
            if len(recent_moves) > 5:
                recent_moves.pop(0)
            if (y, x) in recent_moves[:-1]:
                self.reward -= 0.5  # oscillation penalty

            new_dist = np.linalg.norm([y - gy, x - gx])
            prog = prev_dist - new_dist
            if prog > 0:
                self.reward += prog * 1.5
                no_prog = 0
            else:
                self.reward -= 0.02
                no_prog += 1

            if no_prog > 20:
                self.reward -= 5
                break

            if (y, x) == tuple(goal_pos):
                self.reward += 100 + (max_steps - len(path)) * 0.5
                break

        fin_dist = np.linalg.norm([y - gy, x - gx])
        # Bound extreme accumulated reward to keep scores stable
        self.reward = float(np.clip(self.reward, -200.0, 200.0))
        if (y, x) != tuple(goal_pos):
            if fin_dist < 2:
                self.reward += 30
            elif fin_dist < 5:
                self.reward += 15
            elif fin_dist < 10:
                self.reward += 5

        return path

    def fitness(
        self,
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        epsilon_start: float = EPS_START,
        epsilon_end: float = EPS_END,
        action_noise_std: float = ACTION_NOISE_STD,
    ) -> float:
        path = self.simulate(
            grid,
            start_pos,
            goal_pos,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            action_noise_std=action_noise_std,
        )
        end_y, end_x = path[-1]
        gy, gx = goal_pos
        dist = np.linalg.norm([end_y - gy, end_x - gx])
        unique_cells = len(set(path))
        revisit_penalty = len(path) - unique_cells

        score = 0.0
        score -= dist * 1.0
        score -= len(path) * 0.002
        score -= revisit_penalty * 0.2
        score += unique_cells * 0.1
        score += self.reward

        if (end_y, end_x) == tuple(goal_pos):
            max_steps = grid.shape[0] * grid.shape[1] // 2
            score += 180 + (max_steps - len(path)) * 0.6
        else:
            if dist < 2:
                score += 30
            elif dist < 5:
                score += 15
            elif dist < 10:
                score += 5

        return score

    def get_weights(self):
        return [self.W1.copy(), self.W2.copy()]


def load_metadata() -> pd.DataFrame:
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


def sample_maze(train_df: pd.DataFrame, gen_idx: int, total_gen: int) -> pd.Series:
    if not CURRICULUM:
        return train_df.sample(1).iloc[0]
    sorted_df = train_df.sort_values("path_len")
    if gen_idx < WARMUP_GENS:
        frac = WARMUP_EASY_FRAC
    else:
        ramp = (gen_idx - WARMUP_GENS) / max(1, total_gen - WARMUP_GENS)
        frac = WARMUP_EASY_FRAC + (1 - WARMUP_EASY_FRAC) * min(1.0, ramp)
    end_idx = int(len(sorted_df) * frac)
    end_idx = max(1, min(end_idx, len(sorted_df)))
    return sorted_df.iloc[random.randint(0, end_idx - 1)]


def mutate(agent: SwarmAgent, cfg: EAConfig) -> SwarmAgent:
    w1, w2 = agent.get_weights()
    mask1 = np.random.rand(*w1.shape) < cfg.mutation_rate
    mask2 = np.random.rand(*w2.shape) < cfg.mutation_rate
    w1 = w1 + mask1 * np.random.randn(*w1.shape) * cfg.mutation_std
    w2 = w2 + mask2 * np.random.randn(*w2.shape) * cfg.mutation_std
    if cfg.weight_clip is not None and cfg.weight_clip > 0:
        w1 = np.clip(w1, -cfg.weight_clip, cfg.weight_clip)
        w2 = np.clip(w2, -cfg.weight_clip, cfg.weight_clip)
    return SwarmAgent(weights=[w1, w2])


def crossover(parent_a: SwarmAgent, parent_b: SwarmAgent) -> SwarmAgent:
    w1a, w2a = parent_a.get_weights()
    w1b, w2b = parent_b.get_weights()
    mask1 = np.random.rand(*w1a.shape) < 0.5
    mask2 = np.random.rand(*w2a.shape) < 0.5
    w1 = np.where(mask1, w1a, w1b)
    w2 = np.where(mask2, w2a, w2b)
    return SwarmAgent(weights=[w1, w2])


def evaluate_agent(agent: SwarmAgent, mazes: List[np.ndarray], cfg: EAConfig) -> float:
    scores = []
    for grid in mazes:
        start_pos = tuple(np.argwhere(grid == CELL_START)[0])
        goal_pos = tuple(np.argwhere(grid == CELL_GOAL)[0])
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


def train_ea(train_df: pd.DataFrame, cfg: EAConfig):
    population = [SwarmAgent(hidden_size=cfg.hidden_size) for _ in range(cfg.pop_size)]
    best_agent = None
    best_score = -np.inf

    for gen in range(cfg.generations):
        # Curriculum: sample a batch of mazes to evaluate this generation
        mazes = []
        for _ in range(cfg.mazes_per_gen):
            rec = sample_maze(train_df, gen, cfg.generations)
            grid = np.load(os.path.join(DATA_DIR, rec["filename"]))
            mazes.append(grid)

        scores = [evaluate_agent(agent, mazes, cfg) for agent in population]
        ranked = [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]
        if scores[np.argmax(scores)] > best_score:
            best_score = scores[np.argmax(scores)]
            best_agent = ranked[0]
        elite_count = max(1, int(cfg.elite_frac * cfg.pop_size))
        elites = ranked[:elite_count]
        new_pop = elites.copy()
        while len(new_pop) < cfg.pop_size:
            parent_a, parent_b = random.sample(elites, k=2) if len(elites) > 1 else (elites[0], elites[0])
            if random.random() < cfg.crossover_prob:
                base_child = crossover(parent_a, parent_b)
            else:
                base_child = parent_a
            child = mutate(base_child, cfg)
            new_pop.append(child)
        population = new_pop
        if (gen + 1) % 5 == 0 or gen == 0:
            print(
                "[EA] Gen {}/{} best={:.2f} mean={:.2f} median={:.2f} p90={:.2f}".format(
                    gen + 1, cfg.generations, max(scores), np.mean(scores), np.median(scores), np.percentile(scores, 90)
                )
            )
    return best_agent


def eval_split(agent: SwarmAgent, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    subset = df.sample(min(EVAL_MAX, len(df)), random_state=42)
    for _, row in subset.iterrows():
        grid = np.load(os.path.join(DATA_DIR, row["filename"]))
        start_pos = tuple(np.argwhere(grid == CELL_START)[0])
        goal_pos = tuple(np.argwhere(grid == CELL_GOAL)[0])
        path = agent.simulate(grid, start_pos, goal_pos)
        success = path[-1] == tuple(goal_pos)
        steps = len(path)
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
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    meta = load_metadata()
    train_df = meta[meta["split"] == "train"]
    test_df = meta[meta["split"] == "test"]

    # Probe env
    cfg = EAConfig()
    best_agent = train_ea(train_df, cfg)

    # Save weights
    torch.save({"W1": best_agent.W1, "W2": best_agent.W2}, MODEL_PATH)
    print(f"[EA] Saved best agent weights to {MODEL_PATH}")

    # Evaluate
    train_eval = eval_split(best_agent, train_df)
    test_eval = eval_split(best_agent, test_df)

    train_eval.to_csv(os.path.join(METRICS_DIR, "swarm_train_eval.csv"), index=False)
    test_eval.to_csv(os.path.join(METRICS_DIR, "swarm_test_eval.csv"), index=False)

    train_success = train_eval["success"].mean()
    test_success = test_eval["success"].mean()
    train_ratio = train_eval["step_ratio"].dropna()
    test_ratio = test_eval["step_ratio"].dropna()
    print(f"[EA Eval] train_success={train_success:.3f}, test_success={test_success:.3f}")
    if len(train_ratio) > 0 and len(test_ratio) > 0:
        print(
            f"[EA Eval] step_ratio mean (train/test) = {train_ratio.mean():.2f}/{test_ratio.mean():.2f}, "
            f"median = {train_ratio.median():.2f}/{test_ratio.median():.2f}"
        )

    # Visualization of best agent on an easy maze
    sample_rec = train_df.sort_values("path_len").iloc[0]
    grid = np.load(os.path.join(DATA_DIR, sample_rec["filename"]))
    start_pos = tuple(np.argwhere(grid == CELL_START)[0])
    goal_pos = tuple(np.argwhere(grid == CELL_GOAL)[0])
    path = best_agent.simulate(grid, start_pos, goal_pos)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow((grid == CELL_WALL).astype(int), cmap="gray_r", origin="upper")
    sy, sx = start_pos
    gy, gx = goal_pos
    ax.scatter(sx, sy, c="blue", label="Start")
    ax.scatter(gx, gy, c="red", label="Goal")
    ys, xs = zip(*path)
    ax.plot(xs, ys, c="green", alpha=0.7, label="Path")
    ax.set_title("Best Swarm Agent Path (sample maze)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"[EA] Saved sample path plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
