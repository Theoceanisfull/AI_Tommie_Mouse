# AI Tommie Mouse

Reinforcement Learning project where a mouse agent learns to navigate 2D mazes to find cheese. Includes custom Gymnasium environment, PyGame renderer, generated maze dataset with A* ground truth, and several baseline agents.

## Features
- Custom Gymnasium environment with PyGame visualization.
- Egocentric sensing, short turn memory, and velocity-aware walk/run actions.
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
python main.py --mode train --algo ppo
python main.py --mode train --algo dqn
python main.py --mode train --algo ea
python main.py --mode train --algo qlearn
python main.py --mode train --algo imitation
```
Training outputs go to `models/` when available (e.g., `qlearn_training.png`, `imitation_training.png`) and prints evaluation summaries (success rate, step ratio vs. optimal).

3) **Visualize a trained agent** (PyGame)
```bash
python main.py --mode visualize --algo ppo --difficulty Medium
python main.py --mode visualize --algo dqn --difficulty Medium
python main.py --mode visualize --algo ea --difficulty Medium
python main.py --mode visualize --algo qlearn --difficulty Medium
python main.py --mode visualize --algo imitation --difficulty Medium
```

## Environment Details
- **Observation**: length-12 egocentric vector (front/left/right cells: wall=1.0, open=0.0, cheese=0.5), orientation one-hot, last three relative turns, normalized velocity, and a smell signal that strengthens as open-path distance to cheese shrinks.
- **Actions (8 discrete)**: 0-3 walk (up/right/down/left); 4-7 run (up/right/down/left). Running covers more distance but raises crash risk near corners.
- **Rewards**: +100 for reaching cheese; wall penalties (-5 walk / -10 run) with velocity dampening; shaping based on A* distance reduction (olfactory cue) and penalties for deviating from optimal path; speed bonuses for clean straight runs, penalties for rushing into corners.

## Notes on Imitation Learning
- Collects expert (obs, action) pairs by following stored A* shortest paths from `maze_metadata.csv` (train split).
- If `path_file` entries are missing, regenerate data with `python main.py --mode generate` to recreate both mazes and optimal paths.
