import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import os
from maze_env import MazeEnv

# Keep your hyperparameters the same...
EPISODES = 5500 #ed to give more time at higher curriculum levels
MAX_STEPS = 5000
GAMMA = 0.99     # Increased Gamma: In mazes, we care about long-term future
EPSILON = 1.0
EPSILON_DECAY = 0.995 # Slower decay
EPSILON_MIN = 0.05
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 500

# ---------------------------
# Helper: BFS to find distances from Goal
# ---------------------------
def get_distance_map(env):
    """
    Returns a dict distance -> list of (r, c) cells reachable from goal on the current maze.
    Uses the env's goal_pos and wall encoding (0 free, 1 wall).
    """
    rows, cols = env.maze.shape
    goal_pos = env.goal_pos  # set by MazeEnv during init
    dist_map = np.full((rows, cols), -1)
    q = deque([(goal_pos, 0)])
    dist_map[goal_pos] = 0

    by_dist = {}
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while q:
        (r, c), d = q.popleft()
        by_dist.setdefault(d, []).append((r, c))
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and env.maze[nr, nc] == 0 and dist_map[nr, nc] == -1:
                dist_map[nr, nc] = d + 1
                q.append(((nr, nc), d + 1))
    return by_dist

# ---------------------------
# Custom Reset Logic
# ---------------------------
def reset_curriculum(env, difficulty_radius, starts_by_dist):
    """
    Resets the environment but forces the agent to a specific state 
    based on the current curriculum difficulty.
    """
    # 1. Standard reset to clear internal flags
    state, info = env.reset()
    
    # 2. Pick a start distance. 
    # We allow any distance from 1 up to current 'difficulty_radius'
    # This prevents 'catastrophic forgetting' of easier tasks.
    actual_dist = np.random.randint(1, difficulty_radius + 1)
    
    # If we don't have cells at exactly this distance, clip to max available
    max_avail_dist = max(starts_by_dist.keys())
    actual_dist = min(actual_dist, max_avail_dist)
    
    # 3. Pick a random cell at that distance
    options = starts_by_dist.get(actual_dist, [])
    if not options:
        # Fallback if dictionary lookup fails
        start_pos = (0, 0)
    else:
        start_pos = random.choice(options)
    
    # 4. Force the agent position in the env
    env.agent_pos = start_pos
    env.current_steps = 0  # ensure step counter is consistent

    # Recalculate the integer state for the agent input (r * n_cols + c)
    state_idx = start_pos[0] * env.n_cols + start_pos[1]
    return state_idx, {"position": start_pos}

# ---------------------------
# Model definitions
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

def one_hot_state(state, n_states):
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v

# ---------------------------
# Setup / Training (only when run as a script)
# ---------------------------
if __name__ == "__main__":
    # Use a valid hard maze from the generated 31x31 set
    DATA_MAZE_PATH = "../src/data/3131/maze_1.npy"
    DATA_DIR = "../src/data/3131"
    maze = np.load(DATA_MAZE_PATH)
    if 2 not in maze or 3 not in maze:
        raise ValueError(f"Maze {DATA_MAZE_PATH} must contain start=2 and goal=3 markers for eval/GIF.")
    env = MazeEnv(maze, render_mode=None)

    # Pre-calculate distances for curriculum
    valid_starts_by_dist = get_distance_map(env)
    max_maze_dist = max(valid_starts_by_dist.keys())

    # Curriculum settings
    current_difficulty = 2  # start close to the goal, will ramp up
    success_buffer = deque(maxlen=20)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    base_max_dist = max_maze_dist

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    episode_rewards = []
    start_time = time.time()

    # ---------------------------
    # Training Loop
    # ---------------------------
    steps = 0
    for episode in range(EPISODES):
        
        # --- CURRICULUM RESET (single maze) ---
        state, _ = reset_curriculum(env, current_difficulty, valid_starts_by_dist)
        state_vec = one_hot_state(state, n_states)
        
        total_reward = 0
        done = False
        terminated = False
        truncated = False
        
        # Scale max steps based on current difficulty for this maze
        current_max_steps = min(MAX_STEPS, min(current_difficulty, max_maze_dist) * 5 + 50)

        for step in range(current_max_steps):
            steps += 1
            
            # Action Selection
            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state_vec))
                    action = torch.argmax(q_vals).item()

            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_vec = one_hot_state(next_state, n_states)
            
            # Custom done logic for curriculum
            # If we hit the goal, terminated is True
            done = terminated or truncated

            # Store
            memory.append((state_vec, action, reward, next_state_vec, done))
            state_vec = next_state_vec
            state = next_state
            total_reward += reward

            # Train
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
                
                states_t = torch.tensor(np.stack(states_b), dtype=torch.float32)
                actions_t = torch.tensor(actions_b, dtype=torch.int64).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1)
                next_states_t = torch.tensor(np.stack(next_states_b), dtype=torch.float32)
                dones_t = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1)

                q_vals = policy_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                    q_target = rewards_t + GAMMA * next_q * (1 - dones_t)
                
                loss = nn.MSELoss()(q_vals, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
                
        # --- CURRICULUM UPDATE ---
        # Did we solve it? (Assuming +100 or positive reward means goal reached)
        is_success = 1 if terminated else 0
        success_buffer.append(is_success)
        
        # If we have > 80% success rate over last 20 episodes, increase difficulty
        success_rate = sum(success_buffer) / len(success_buffer)
        if len(success_buffer) >= 10 and success_rate > 0.8:
            if current_difficulty < base_max_dist:
                current_difficulty += 1
                success_buffer.clear() # Reset buffer for new difficulty
                print(f"*** Level Up! Increasing Difficulty to Radius {current_difficulty} ***")

        # Decay Epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        episode_rewards.append(total_reward)

        if episode % 20 == 0:
            elapsed = (time.time() - start_time) / 60.0
            mem_len = len(memory)
            print(f"Ep {episode} | Diff: {current_difficulty}/{max_maze_dist} | Success: {success_rate:.2f} | Eps: {EPSILON:.3f} | Steps: {steps} | Replay: {mem_len} | Elapsed: {elapsed:.1f} min")

    # ---------------------------
    # Save model
    # ---------------------------
    import os
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("../trained_models", exist_ok=True)
    torch.save(policy_net.state_dict(), "../trained_models/dqn_model_curriculum.pth")
    print("Saved curriculum DQN model to ../trained_models/dqn_model_curriculum.pth")

    # ---------------------------
    # Save reward curve
    # ---------------------------
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Curriculum Training Progress")
    plt.grid(True)
    plt.savefig("metrics/dqn_rewards_curriculum.png")
    plt.close()
    print("Saved reward plot to metrics/dqn_rewards_curriculum.png")

    # ---------------------------
    # Single greedy eval with visualization (GIF)
    # ---------------------------
    import imageio.v2 as imageio
    import pygame

    policy_net.eval()
    viz_maze = maze  # reuse the training maze for a deterministic eval

    eval_env = MazeEnv(viz_maze, render_mode="human")
    state, info = eval_env.reset()
    state_vec = one_hot_state(state, n_states)
    eval_env.render()  # ensure pygame surface exists before capturing
    frames = []
    terminated = False
    truncated = False

    # Limit eval steps so we don't hang the eval phase
    max_eval_steps = min(MAX_STEPS, eval_env.max_steps)
    for step in range(max_eval_steps):
        with torch.no_grad():
            q_vals = policy_net(torch.tensor(state_vec))
            action = torch.argmax(q_vals).item()

        state, reward, terminated, truncated, info = eval_env.step(action)
        state_vec = one_hot_state(state, n_states)

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

    eval_env.close()

    if frames:
        gif_path = "metrics/dqn_run_curriculum.gif"
        imageio.mimsave(gif_path, frames, fps=10, format="GIF")
        print(f"Saved evaluation run animation to {gif_path}")
    else:
        print("No frames captured during eval (pygame surface not available).")

    if not terminated:
        print(f"WARNING: Eval did not reach goal (terminated={terminated}, truncated={truncated}); GIF shows failure loop.")
    print(f"Eval finished after {step+1} steps; success={terminated}, truncated={truncated}")
    print(f"Final difficulty reached: {current_difficulty}/{max_maze_dist}")
