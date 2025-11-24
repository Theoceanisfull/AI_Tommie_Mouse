# AI Tommie Mouse

Reinforcement Learning project where a mouse agent learns to navigate 2D mazes to find cheese. Includes custom Gymnasium environment, PyGame renderer, generated maze dataset with A* ground truth, and several baseline agents.

## Features
- Custom Gymnasium environments:
  - `mouse` (default): PyGame rendering, egocentric sensing, turn memory, walk/run dynamics, smell shaping toward cheese.
  - `simple`: 4-action walls/direction-to-goal observation, designed for evolutionary search.
- 4,000 generated mazes (easy/medium/hard; perfect + imperfect) with A* optimal paths and lengths; train/test split baked into metadata.
- Baseline algorithms: PPO, DQN, Evolutionary Algorithm, Imitation Learning (A* demonstrations), and tabular Q-Learning.

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
1) **Generate data** (creates `data/*.npy` and `maze_metadata.csv` with optimal paths)
```bash
python main.py --mode generate
```

2) **Train a model** (headless for speed)
```bash
python main.py --mode train --algo ppo --env mouse
python main.py --mode train --algo dqn --env mouse
python main.py --mode train --algo ea           # defaults to --env simple
python main.py --mode train --algo qlearn --env mouse
python main.py --mode train --algo imitation --env mouse
```
Training outputs go to `models/` when available (e.g., `qlearn_training.png`, `imitation_training.png`) and prints evaluation summaries (success rate, step ratio vs. optimal).

3) **Visualize a trained agent** (PyGame)
```bash
python main.py --mode visualize --algo ppo --difficulty Medium --env mouse
python main.py --mode visualize --algo dqn --difficulty Medium --env mouse
python main.py --mode visualize --algo ea  --difficulty Medium --env simple   # EA checkpoints assume simple by default
python main.py --mode visualize --algo qlearn --difficulty Medium --env mouse
python main.py --mode visualize --algo imitation --difficulty Medium --env mouse
```

## Environment Details
- **mouse env**: length-12 egocentric observation (front/left/right cells, orientation one-hot, turn memory, normalized velocity, smell signal), 8 actions (walk/run in 4 directions), rewards with cheese bonus, wall penalties, smell/optimal-path shaping, and speed nuances.
- **simple env**: 6-D observation (walls up/right/down/left, normalized goal direction), 4 actions (up/right/down/left), sparse shaping toward goal, lighter dynamics tailored for evolutionary search.

## Notes on Imitation Learning
- Collects expert (obs, action) pairs by following stored A* shortest paths from `maze_metadata.csv` (train split).
- If `path_file` entries are missing, regenerate data with `python main.py --mode generate` to recreate both mazes and optimal paths.
