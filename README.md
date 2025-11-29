# AI Tommie Mouse

Reinforcement Learning project where a mouse agent learns to navigate 2D mazes to find cheese. Includes custom Gymnasium environment, PyGame renderer, generated maze dataset with A* ground truth, and several baseline agents.

## Features
- Custom Gymnasium environments:
  - `mouse` (default): PyGame rendering, egocentric sensing, turn memory, walk/run dynamics, smell shaping toward cheese.
  - `simple`: 4-action walls/direction-to-goal observation, designed for evolutionary search.
- 4,000 generated mazes (easy/medium/hard; perfect + imperfect) with A* optimal paths and lengths; train/test split baked into metadata.
- Baseline algorithms: PPO, DQN, Evolutionary Algorithm, Imitation Learning (A* demonstrations), and tabular Q-Learning.
- Standalone Q-Learning/DQN experiments with curriculum, single-maze and multi-maze trainers, and evaluation GIF/plots.

## Quick Start
```bash
git clone https://github.com/Theoceanisfull/AI_Tommie_Mouse.git
cd AI_Tommie_Mouse/rl_mouse_maze

# Optional: create a branch for your work
git checkout -b my-feature

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1) **Generate data** (creates per-difficulty folders under `src/data/` with `maze_metadata.csv`)
```bash
python main.py --mode generate
# or using the standalone generator (also builds per-difficulty folders: 11x11, 21x21, 31x31)
python src/maze_generator.py
```

2) **Train a model** (headless for speed)
```bash
python main.py --mode train --algo ppo --env mouse
python main.py --mode train --algo dqn --env mouse
python main.py --mode train --algo ea           # defaults to --env simple
python main.py --mode train --algo qlearn --env mouse
python main.py --mode train --algo imitation --env mouse

# Standalone trainers (Q-Learning folder; metrics stored in Q-Learning/metrics, models in trained_models/)
# Single-maze DQN curriculum on a fixed maze (../data/maze_755.npy)
python Q-Learning/DQN_Single.py
# Multi-maze DQN curriculum on Medium (21x21) mazes using metadata in data/2121 (90/10 split)
python Q-Learning/DQN_Multi.py
```
Training outputs go to `trained_models/` when available (e.g., `qlearn_training.png`, `imitation_training.png`) and prints evaluation summaries (success rate, step ratio vs. optimal). Each trainer folder (e.g., `Q-Learning/`) has its own `metrics/` for plots/GIFs.

3) **Visualize a trained agent** (PyGame)
```bash
python main.py --mode visualize --algo ppo --difficulty Medium --env mouse
python main.py --mode visualize --algo dqn --difficulty Medium --env mouse
python main.py --mode visualize --algo ea  --difficulty Medium --env simple   # EA checkpoints assume simple by default
python main.py --mode visualize --algo qlearn --difficulty Medium --env mouse
python main.py --mode visualize --algo imitation --difficulty Medium --env mouse

# Standalone inference / GIF creation (writes to Q-Learning/metrics by default)
python Q-Learning/test_dqn.py --render --maze ../data/maze_757.npy --gif metrics/dqn_test.gif
python Q-Learning/test_dqn.py --batch ../data/maze_757.npy ../data/maze_755.npy
```

## Environment Details
- **mouse env**: length-12 egocentric observation (front/left/right cells, orientation one-hot, turn memory, normalized velocity, smell signal), 8 actions (walk/run in 4 directions), rewards with cheese bonus, wall penalties, smell/optimal-path shaping, and speed nuances.
- **simple env**: 6-D observation (walls up/right/down/left, normalized goal direction), 4 actions (up/right/down/left), sparse shaping toward goal, lighter dynamics tailored for evolutionary search.

## Notes on Imitation Learning
- Collects expert (obs, action) pairs by following stored A* shortest paths from `maze_metadata.csv` (train split).
- If `path_file` entries are missing, regenerate data with `python main.py --mode generate` to recreate both mazes and optimal paths.

## Standalone DQN/Q-Learning (Q-Learning folder)
- `DQN_Single.py`: curriculum DQN on a single 21x21 maze (`../data/maze_755.npy`), saves model + reward plot + eval GIF.
- `DQN_Multi.py`: curriculum DQN across the Medium dataset (`data/2121`, 90/10 split), saves model, reward/success plots, test metrics CSV, and eval GIF.
- `test_dqn.py`: load a saved DQN and run a greedy rollout on any maze, with optional render and GIF export.

## Repo Structure (recommended)
- `data/` (and per-difficulty subfolders like `data/2121/`)
- `trained_models/` (all saved weights/checkpoints)
- `Q-Learning/` (tabular + DQN experiments, with `metrics/`)
- `Imitation-Learning/` (imitation-specific scripts/tests, with `metrics/`)
- `EA/` (evolutionary scripts/tests, with `metrics/`)
- `PPO/` (ppo/dqn classic runs if separated, with `metrics/`)
- `src/` (shared code: envs, agents, main CLI)
