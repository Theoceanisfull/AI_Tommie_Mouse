<<<<<<< HEAD
# AI_Tommie_Mouse
=======
RL Mouse Maze Project

A Reinforcement Learning environment where a mouse agent learns to traverse 2D mazes to find cheese.

Features

Custom Environment: Built on Gymnasium with PyGame rendering.

Dynamics: Agent has egocentric sensing with limited turn memory plus a velocity control (walk vs. run). Running covers more ground but increases crash risk, especially when approaching corners.

Dataset: 4000 Generated mazes (2000 Perfect, 2000 Imperfect) across Easy, Medium, and Hard difficulties.

Ground Truth: A* calculated optimal paths and lengths stored alongside each maze, with an 80/20 train/test split baked into the metadata.

Algorithms:

PPO (Proximal Policy Optimization)

DQN (Deep Q-Network)

Evolutionary Algorithm

Imitation Learning (uses A* shortest-path demonstrations)

Tabular Q-Learning (simple baseline for env sanity-check)

Installation

Create a virtual environment:

python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows


Install dependencies:

pip install -r requirements.txt

Getting the code and creating your own branch

git clone <repo-url>
cd rl_mouse_maze
git checkout -b your-feature-branch
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

After that, you can edit `src/agents.py` to add your own model or training routine, then run the commands below to train and visualize it.


Usage

1. Generate the Data

Before training, you must generate the maze dataset. This creates .npy files in data/ and a maze_metadata.csv containing ground truth.

python main.py --mode generate


2. Train a Model

Train an agent on the training split. The visualization is disabled during training for speed.

python main.py --mode train --algo ppo
python main.py --mode train --algo dqn
python main.py --mode train --algo ea
python main.py --mode train --algo qlearn
python main.py --mode train --algo imitation

Each training run writes curves/metrics to `models/` when available:
- Q-learning: `qlearn_training.png` with reward, cumulative success rate, and average step ratio (agent steps / optimal steps).
- Imitation: `imitation_training.png` with supervised loss per epoch.
- All algos print a quick evaluation summary after training (success rate and average step ratio on a held-out maze when available).


3. Visualize Results

Run the trained model in "Video Game" mode with PyGame visualization.

python main.py --mode visualize --algo ppo --difficulty Medium
python main.py --mode visualize --algo dqn --difficulty Medium
python main.py --mode visualize --algo ea --difficulty Medium
python main.py --mode visualize --algo qlearn --difficulty Medium
python main.py --mode visualize --algo imitation --difficulty Medium

Notes on imitation learning:
- The imitation trainer collects expert (obs, action) pairs by walking the stored A* shortest paths from `maze_metadata.csv` (train split only).
- Make sure you regenerate data (`--mode generate`) if your metadata is missing `path_file` entries; the generator writes both maze files and optimal paths.

Training visualization and metrics

- Q-learning training logs reward per episode, cumulative success rate (fraction of episodes reaching the cheese), and average step ratio (agent steps / optimal steps). Lower step ratios are better; 1.0 is optimal.
- Imitation training logs supervised loss per epoch.
- Post-training evaluation for all algorithms reports:
  - Success rate: how often the agent reaches the cheese over a small eval set.
  - Step ratio: average (agent steps / optimal steps). Ratios >1 indicate longer-than-optimal paths; e.g., 1.0 is perfect, 2.0 is twice the optimal length.


Environment Details

Observation: Egocentric vector of length 12:
front/left/right cells (blocked/wall=1.0, open=0.0, cheese=0.5), orientation one-hot, last three relative turns (left/straight/right/about-face), current normalized velocity, and a smell intensity signal that only grows when an unobstructed path to the cheese shrinks.

Actions: 8 Discrete actions.

0-3: Walk (Up, Right, Down, Left) - Moves 1 cell and decelerates.

4-7: Run (Up, Right, Down, Left) - Accelerates and attempts 2 cells; excessive speed near corners increases crash penalties.

Rewards:

+100 for reaching Cheese.
Wall penalties: -5 (walk) / -10 (run) with velocity dampening.
Shaping: Positive when open-path (A*) distance to cheese shrinks (olfactory cue), neutral when following the optimal path, increasing penalties as movement deviates from the optimal route.
Speed nuance: Bonus for running straight corridors, penalty when rushing into corners.
>>>>>>> b5eafdbe (Initial commit)
