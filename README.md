# AI Tommie Mouse
RL playground for solving procedurally generated mazes. Ships a maze generator with A* ground truth, custom Gymnasium environments, and multiple learning families: tabular/deep Q-learning, PPO, imitation (LSTM/MLP/ViT/PPO-BC), evolutionary strategies, and a swarm-inspired EA.

## Setup
- Python 3.10+. Create venv: `python -m venv venv && source venv/bin/activate` (Win: `venv\Scripts\activate`).
- Install: `pip install -r requirements.txt`.


## Maze Data (src/maze_generator.py)
- 4k mazes per difficulty: Easy 11x11 (`1111`), Medium 21x21 (`2121`), Hard 31x31 (`3131`); 90/10 train/test split in `maze_metadata.csv`.
- Markers: start=2, goal=3. Stored files: `maze_*.npy`, `*_path.npy` (A*), optional `*_dist.npy` / `*_actions.npy` flow fields.
- Regenerate all: `python main.py --mode generate`.

## Environments (src/mouse_env.py)
- **MouseMazeEnv** (`env=mouse`): turn-and-move FoV; actions forward/turn L/R/back. Obs: whiskers + forward cone + smell + last two actions. Rewards: step/collision penalties, inverse-distance shaping, goal bonus.
- **MouseMazeEnvLegacy**: straight-move FoV (Up/Right/Down/Left); obs: 8-tile FoV + orientation + turn memory (+ optional smell); revisit/optimal-path penalties available. Used by older DQN/imitation scripts.
- **SimpleMazeEnv** (`env=simple`): 6-D obs (walls + normalized goal vector), 4 actions; light shaping + shortest-path action bonus. Default for EA/tabular/DQN baselines.
- **FOVBasicEnv**: minimal FoV + orientation; light shaping. PyGame rendering; set `SDL_VIDEODRIVER=dummy` if headless.

## Repository Map
- `main.py` – CLI for generate/train/visualize/evaluate.
- `src/` – environments, agents, maze generator, datasets.
- `Q-Learning/`, `Imitation-Learning/`, `Swarm-Learning/` – standalone experiment folders with their own `metrics/`.
- `trained_models/` – saved weights/checkpoints.
- `metrics/` + per-folder metrics – plots/CSVs/GIFs.

## Directory Guides
### Q-Learning/
- **Scripts**:  
  - `DQN_Single.py` – curriculum DQN on one hard maze (`src/data/3131/maze_1.npy`).  
  - `DQN_Multi.py` – curriculum DQN across Medium mazes (90/10 split).  
  - `DQN_Multi_Improved.py` – recurrent DQN (embed + LSTM) with sequence replay, stall/oscillation penalties.  
  - `DDQN_R2D2.py` – double-DQN-style recurrent agent with sequence sampling.  
  - `tab_qlearning.py` / `q_learning.py` – tabular baselines.  
  - `test_dqn.py` – roll out saved DQN(s); optional render/GIF.
- **Run**:  
  - `python Q-Learning/DQN_Single.py`  
  - `python Q-Learning/DQN_Multi.py`  
  - `python Q-Learning/DQN_Multi_Improved.py`  
  - `python Q-Learning/DDQN_R2D2.py`  
  - `python Q-Learning/tab_qlearning.py`  
  - `python Q-Learning/test_dqn.py --maze ../src/data/2121/maze_757.npy --gif metrics/dqn_test.gif`
- **Outputs**: models in `trained_models/` (e.g., `dqn_model_multi.pth`); reward/success plots, CSVs, GIFs in `Q-Learning/metrics/`. Uses `MouseMazeEnvLegacy` unless noted.

### Imitation-Learning/
- **Scripts**:  
  - `imitation_single.py` – BC on one hard maze; loss plot + GIF.  
  - `imitation_multi.py` – BC on 90% Hard mazes + optional REINFORCE fine-tune; supports flow-field `*_actions.npy`.  
  - `imitation_vit.py` – ViT imitation with optional on-policy adaptation at eval.  
  - `ppo_bc.py` – PPO warm-started with behavior cloning (SB3).  
  - `RW_ETL.ipynb` – data/metrics exploration.
- **Run**:  
  - `python Imitation-Learning/imitation_single.py`  
  - `python Imitation-Learning/imitation_multi.py`  
  - `python Imitation-Learning/imitation_vit.py`  
  - `python Imitation-Learning/ppo_bc.py`
- **Outputs**: models in `trained_models/` (`imitation_mouse.pth`, `imitation_multi.pth`, `imitation_vit.pth`, etc.); plots/CSVs/GIFs in `Imitation-Learning/metrics/`. Uses `MouseMazeEnvLegacy`.

### Swarm-Learning/
- **Scripts**:  
  - `swarm_ea.py` – population EA on Hard mazes; curriculum over path length, crossover + Gaussian mutation + weight clipping.  
  - `swarm_single.py` – simplified single-maze EA.  
  - `metrics/` – CSVs and path plots (e.g., `swarm_best_path.png`).
- **Run**:  
  - `python Swarm-Learning/swarm_ea.py`  
  - `python Swarm-Learning/swarm_single.py`
- **Outputs**: `trained_models/swarm_ea_best.pth` (and `swarm_single_best.pth`); metrics/plots in `Swarm-Learning/metrics/`.

## Algorithm Cheat Sheet
- **Tabular Q-learning** (`src/agents.py::QLearningAgent`, `tab_qlearning.py`): epsilon-greedy table; TD update `Q ← Q + lr*(r + γ max_a' Q' - Q)`.
- **Deep Q-learning**: MLP (obs→128→128→actions) with replay + target net; linear epsilon decay. Recurrent variants add LSTM + sequence replay (burn-in/seq_len). SB3 DQN via `agents.get_dqn_model`.
- **PPO**: Actor-critic MLP (64,64) with GAE, clipped objective, entropy/value losses. SB3 PPO via `agents.get_ppo_model`.
- **Imitation**: LSTM policy (embed→LSTM→head) with CE loss, Adam, scheduler, grad clip; MLP alternative; ViT variant (patchify → Transformer → head); PPO+BC hybrid (`ppo_bc.py`).
- **Evolutionary**: `SimpleEvolutionaryAgent` (embed→1-step LSTM→head) evolved by ranking, elites, lucky survivors, Gaussian mutation (fitness = avg episode return); swarm EA uses tiny MLP (goal delta + walls + visited), shaped fitness, crossover + mutation + optional curriculum.

## Tips & Pitfalls
- Ensure mazes include start=2 and goal=3 (generator enforces).
- For headless servers, set `SDL_VIDEODRIVER=dummy` before scripts that render.
- Reward shaping and epsilon schedules drive exploration; if agents loop/stall, tune step/crash penalties, epsilon decay, and max steps.
- Regenerating data overwrites metadata and can take time; use shipped datasets unless you need new flow fields.
