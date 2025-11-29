# AI Tommie Mouse

Mouse-in-a-maze RL playground with custom Gymnasium environments, PyGame renderers, a generated maze corpus with A* ground truth, and multiple baselines (PPO/DQN/EA/tabular Q-learning/imitation). Data and code are organized to keep experiments isolated (per-training-type `metrics/`, shared `trained_models/`, shared `src/data`).

## Highlights
- **Environments**
  - `mouse` (legacy straight-move): 4 actions (Up/Right/Down/Left), 8-tile FoV + orientation + turn memory (+ optional smell). Start=2, goal=3 must exist. PyGame render uses purple walls, 🐭 agent, 🧀 goal.
  - `simple`: 4 actions, 6-D obs (walls up/right/down/left + normalized goal direction), tuned for EA/tabular/Q-learning/DQN.
- **Data**: 4,000 mazes per difficulty (easy 11x11, medium 21x21, hard 31x31; perfect + imperfect) under `src/data/{1111,2121,3131}` with `maze_metadata.csv` and A* paths (`*_path.npy`).
- **Baselines**: PPO, DQN, Evolutionary Algorithm, tabular Q-learning, Imitation Learning (A* demos). Standalone curriculum DQN scripts for single- and multi-maze training.
- **Artifacts**: Each trainer writes plots/GIFs to its own `metrics/` folder and saves weights to `trained_models/`.

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
- **mouse env (legacy straight-move)**: 8-tile FoV + orientation one-hot + turn memory (+ optional smell); 4 actions. Start=2, goal=3 required. Rewards: goal bonus, wall penalty, shaping toward goal. Purple-wall/emoji render.
- **simple env**: 6-D observation (walls, normalized goal direction), 4 actions, sparse shaping, tuned for EA/tabular/DQN stability.
- Rendering: PyGame with purple walls, 🐭 agent, 🧀 goal; ensure a display is available or use a headless backend if needed.

## Notes on Imitation Learning
- Uses A* paths stored in `maze_metadata.csv` (`path_file` column) to collect (obs, action) pairs with `MouseMazeEnvLegacy`.
- If any `path_file` is missing, regenerate data with `python main.py --mode generate` or `python src/maze_generator.py` to recreate mazes and optimal paths.
- `imitation_single.py` trains/evals on one hard maze (31x31) and writes loss plot + eval GIF to `Imitation-Learning/metrics/`.
- `imitation_multi.py` trains on 90% of `src/data/3131`, tests on 10%; saves loss, eval CSVs (train/test), and eval plots to `Imitation-Learning/metrics/`, model to `trained_models/imitation_multi.pth`.

## Standalone DQN/Q-Learning (Q-Learning folder)
- `DQN_Single.py`: curriculum DQN on a single hard maze (`../src/data/3131/maze_1.npy`), saves model + reward plot + eval GIF (warns if goal not reached).
- `DQN_Multi.py`: curriculum DQN across the Medium dataset (`src/data/2121`, 90/10 split), saves model, reward/success plots, test metrics CSV, and eval GIF.
- `test_dqn.py`: load a saved DQN and run a greedy rollout on any maze, with optional render and GIF export.

## Troubleshooting / Nuances
- **Start/Goal markers**: all envs expect start=2, goal=3. Scripts now raise if missing (prevents silent looping GIFs).
- **Headless PyGame**: if renders fail on a server, set `SDL_VIDEODRIVER=dummy` before running visualization scripts; GIF capture needs a surface.
- **Imitation agent**: 2-layer LSTM with LR scheduler (ReduceLROnPlateau), grad clipping, dropout on the feedforward and LSTM (only when num_layers>1).
- **Curriculum DQN**: both single and multi use one-hot state MLPs (256-128 heads). `DQN_Single` evaluates on the training maze for a deterministic GIF and warns if the goal is not reached.
- **Data splits**: `maze_metadata.csv` carries a 90/10 train/test split; rerun the generator if missing. Paths are relative to `src/data/{1111,2121,3131}`.

## Repo Structure (recommended)
- `src/data/{1111,2121,3131}`: generated mazes + metadata + optimal paths
- `trained_models/`: all saved weights/checkpoints
- `Q-Learning/`: tabular + DQN experiments, with `metrics/`
- `Imitation-Learning/`: imitation scripts/tests, with `metrics/`
- `EA/`, `PPO/`: evolutionary and PPO baselines (each with `metrics/` if used)
- `src/`: shared code (envs, agents, main CLI)
