import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
import os
import imageio.v2 as imageio
import pygame


# ---------------------------
# 1. Define hyperparameters
# ---------------------------
# Longer training + slower epsilon decay to keep exploring harder mazes
EPISODES = 4000          # total training episodes
MAX_STEPS = None         # set after env creation
ALPHA = 0.1              # learning rate
GAMMA = 0.95             # discount factor (value next state more)
EPSILON = 1.0            # exploration probability
EPSILON_DECAY = 0.999    # slower decay to keep exploration
EPSILON_MIN = 0.02       # allow occasional exploration late

# Ensure metrics directory exists
METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------
# 2. Create environment (Medium perfect maze from dataset; harder than outer-wall trivial paths)
# ---------------------------
maze = np.load("../data/maze_755.npy")  # Medium perfect maze (21x21, start=2, goal=3)

env = MazeEnv(maze, render_mode=None)
if MAX_STEPS is None:
    MAX_STEPS = env.max_steps

n_states = env.observation_space.n
n_actions = env.action_space.n

# ---------------------------
# 3. Initialize Q-table
# ---------------------------
Q = np.zeros((n_states, n_actions))

# For logging
episode_rewards = []

# ---------------------------
# 4. Training loop
# ---------------------------
for episode in range(EPISODES):
    state, info = env.reset()
    total_reward = 0
    done = False
    visit_counts = {}
    prev_state = None

    for step in range(MAX_STEPS):
        # ε-greedy action selection
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(Q[state, :])     # exploit best known action

        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Penalize backtracking and lingering in loops
        if prev_state is not None and next_state == prev_state:
            reward -= 0.2  # discourage immediate back-and-forth
        visit_counts[next_state] = visit_counts.get(next_state, 0) + 1
        if visit_counts[next_state] > 3:
            reward -= 0.05 * (visit_counts[next_state] - 3)

        # Q-learning update
        best_next_action = np.argmax(Q[next_state, :])
        td_target = reward + GAMMA * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += ALPHA * td_error

        total_reward += reward
        prev_state = state
        state = next_state

        if done:
            break

    # ε-decay (explore less over time)
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    # Optional progress printing
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}/{EPISODES} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {EPSILON:.3f}")

# ---------------------------
# 5. Single test run with visualization
# ---------------------------
frames = []
env.render_mode = "human"
state, info = env.reset()
done = False
for step in range(MAX_STEPS):
    action = np.argmax(Q[state, :])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # Capture frame if pygame display exists
    surf = None
    try:
        surf = pygame.display.get_surface()
    except Exception:
        surf = None
    if surf is not None:
        frame = pygame.surfarray.array3d(surf).swapaxes(0, 1)
        frames.append(frame)
    if done:
        break
env.close()

if frames:
    video_path = os.path.join(METRICS_DIR, "q_learning_run.gif")
    imageio.mimsave(video_path, frames, fps=10, format="GIF")
    print(f"Saved run animation to {video_path}")

# ---------------------------
# 6. Save Q-table
# ---------------------------
qtable_path = os.path.join(METRICS_DIR, "q_table.csv")
np.savetxt(qtable_path, Q, delimiter=",")
print(f"Saved Q-table to {qtable_path}")

# ---------------------------
# 7. Plot training performance
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Progress")
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(METRICS_DIR, "training_rewards.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved training plot to {plot_path}")
