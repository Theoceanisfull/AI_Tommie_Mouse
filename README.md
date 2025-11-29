# AI Tommie Mouse

Reinforcement Learning project where a mouse agent learns to navigate 2D mazes to find cheese. Includes custom Gymnasium environment, PyGame renderer, generated maze dataset with A* ground truth, and several baseline agents.

## Features
- Custom Gymnasium environments:
  - `mouse`: PyGame rendering, egocentric FoV, turn memory, smell signal; legacy straight-move control used by imitation scripts.
  - `simple`: 4-action, walls + goal-direction observation; tuned for EA/tabular/Q-learning/DQN baselines.
- 4,000 generated mazes per difficulty (easy 11x11, medium 21x21, hard 31x31; perfect + imperfect) with A* optimal paths and lengths; stored under `src/data/{1111,2121,3131}` with `maze_metadata.csv`.
- Baseline algorithms: PPO, DQN, Evolutionary Algorithm, Imitation Learning (A* demonstrations), tabular Q-Learning; curriculum DQN scripts.
- Standalone trainers produce plots/GIFs under their local `metrics/` and save weights to `trained_models/`.

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
# Single-maze DQN curriculum on a fixed hard maze (../src/data/3131/maze_1.npy)
python Q-Learning/DQN_Single.py
# Multi-maze DQN curriculum on Medium (21x21) mazes using metadata in data/2121 (90/10 split)
python Q-Learning/DQN_Multi.py
# Tabular Q-learning baseline and test runner
python Q-Learning/q_learning.py
python Q-Learning/test_dqn.py --render --maze ../src/data/2121/maze_757.npy --gif metrics/dqn_test.gif

# Imitation learning (A* demonstrations on hard mazes)
python Imitation-Learning/imitation_single.py   # single 31x31 maze, saves GIF/metrics/model
python Imitation-Learning/imitation_multi.py    # trains on 90% of src/data/3131, tests on 10%, saves metrics/plots/model
```
Training outputs go to `trained_models/` and each trainer’s local `metrics/` (plots, CSVs, GIFs). Scripts print evaluation summaries (success rate, step ratio vs optimal).

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
- **mouse env (legacy straight-move)**: 8-tile FoV + orientation one-hot + turn memory + optional smell; 4 actions (Up/Right/Down/Left); start=2, goal=3 markers required.
- **simple env**: 6-D observation (walls up/right/down/left, normalized goal direction), 4 actions (up/right/down/left), sparse shaping toward goal, lighter dynamics tailored for evolutionary search.
- Custom renderers: purple walls, mouse/cheese icons (pygame); start=2, goal=3 must exist or scripts will raise.

## Notes on Imitation Learning
- Uses A* paths stored in `maze_metadata.csv` (`path_file` column) to collect (obs, action) pairs with `MouseMazeEnvLegacy`.
- If any `path_file` is missing, regenerate data with `python main.py --mode generate` or `python src/maze_generator.py` to recreate mazes and optimal paths.
- `imitation_single.py` trains/evals on one hard maze (31x31) and writes loss plot + eval GIF to `Imitation-Learning/metrics/`.
- `imitation_multi.py` trains on 90% of `src/data/3131`, tests on 10%; saves loss, eval CSVs (train/test), and eval plots to `Imitation-Learning/metrics/`, model to `trained_models/imitation_multi.pth`.

## Standalone DQN/Q-Learning (Q-Learning folder)
- `DQN_Single.py`: curriculum DQN on a single hard maze (`../src/data/3131/maze_1.npy`), saves model + reward plot + eval GIF (warns if goal not reached).
- `DQN_Multi.py`: curriculum DQN across the Medium dataset (`src/data/2121`, 90/10 split), saves model, reward/success plots, test metrics CSV, and eval GIF.
- `test_dqn.py`: load a saved DQN and run a greedy rollout on any maze, with optional render and GIF export.

## Repo Structure (recommended)
- `src/data/{1111,2121,3131}`: generated mazes + metadata + optimal paths
- `trained_models/`: all saved weights/checkpoints
- `Q-Learning/`: tabular + DQN experiments, with `metrics/`
- `Imitation-Learning/`: imitation scripts/tests, with `metrics/`
- `EA/`, `PPO/`: evolutionary and PPO baselines (each with `metrics/` if used)
- `src/`: shared code (envs, agents, main CLI)
