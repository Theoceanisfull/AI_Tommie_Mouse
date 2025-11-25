import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
import os
import imageio.v2 as imageio
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time


# ---------------------------
# 1. Define hyperparameters
# ---------------------------
EPISODES = 100
MAX_STEPS = 2000
GAMMA = 0.95

EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.02

LR = 0.0005           # learning rate for neural network
BATCH_SIZE = 64       # replay minibatch
MEMORY_SIZE = 20000   # replay buffer size (reduced; still plenty for 21x21 maze)
TARGET_UPDATE = 1000  # steps between target sync


# ---------------------------
# 2. Create environment
# ---------------------------
maze = np.load("../data/maze_755.npy")
env = MazeEnv(maze, render_mode=None)

if MAX_STEPS is None:
    MAX_STEPS = env.max_steps

n_states = env.observation_space.n
n_actions = env.action_space.n


# ---------------------------
# 3. Utilities
# ---------------------------
def one_hot_state(state, n_states):
    """Convert environment integer state into one-hot vector."""
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


# ---------------------------
# 4. Define DQN network
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

    def forward(self, x):
        return self.net(x)


# Initialize networks
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())  # sync weights

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

memory = deque(maxlen=MEMORY_SIZE)

steps = 0
start_time = time.time()
# Announce run immediately so user sees activity
print(
    f"Starting DQN training on maze_755: episodes={EPISODES}, "
    f"max_steps={MAX_STEPS}, replay_size={MEMORY_SIZE}, batch={BATCH_SIZE}",
    flush=True,
)


# ---------------------------
# 5. Replay buffer sample
# ---------------------------
def sample_from_memory(batch_size):
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Stack to numpy first to avoid slow list->tensor creation
    states_np = np.stack(states).astype(np.float32)
    next_states_np = np.stack(next_states).astype(np.float32)

    states      = torch.from_numpy(states_np)
    actions     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.from_numpy(next_states_np)
    dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    return states, actions, rewards, next_states, dones


# ---------------------------
# 6. Training
# ---------------------------
episode_rewards = []

for episode in range(EPISODES):
    state, info = env.reset()
    state_vec = one_hot_state(state, n_states)
    total_reward = 0
    visit_counts = {}
    prev_state = None
    done = False

    for step in range(MAX_STEPS):
        steps += 1

        # ε-greedy action selection
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(state_vec))
                action = torch.argmax(q_vals).item()

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_vec = one_hot_state(next_state, n_states)
        done = terminated or truncated

        # Penalties (same as your Q-learning version)
        if prev_state is not None and next_state == prev_state:
            reward -= 0.2
        visit_counts[next_state] = visit_counts.get(next_state, 0) + 1
        if visit_counts[next_state] > 3:
            reward -= 0.05 * (visit_counts[next_state] - 3)

        # Save experience
        memory.append((state_vec, action, reward, next_state_vec, done))

        state_vec = next_state_vec
        prev_state = next_state
        total_reward += reward

        # ---------------------------
        # DQN update step
        # ---------------------------
        if len(memory) >= BATCH_SIZE:

            states, actions, rewards_b, next_states, dones_b = sample_from_memory(BATCH_SIZE)

            # Q(s,a)
            q_vals = policy_net(states).gather(1, actions)

            # Target: r + γ max_a' Q_target(s', a')
            with torch.no_grad():
                next_q_vals = target_net(next_states).max(dim=1, keepdim=True)[0]
                q_targets = rewards_b + GAMMA * next_q_vals * (1 - dones_b)

            loss = nn.MSELoss()(q_vals, q_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sync target network
        if steps % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # Decay exploration
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if (episode + 1) % 20 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        elapsed = time.time() - start_time
        mins = elapsed / 60.0
        mem_len = len(memory)
        print(
            f"Episode {episode+1}/{EPISODES} | "
            f"AvgReward(100): {avg_reward:.2f} | "
            f"ε={EPSILON:.3f} | "
            f"elapsed={mins:.1f} min | "
            f"steps={steps} | "
            f"replay={mem_len}",
            flush=True,
        )


# ---------------------------
# 7. Save model and plot
# ---------------------------
os.makedirs("metrics", exist_ok=True)
save_path = "metrics/dqn_model.pth"
torch.save(policy_net.state_dict(), save_path)

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Progress")
plt.grid()
plt.savefig("metrics/dqn_rewards.png")
plt.close()
total_mins = (time.time() - start_time) / 60.0
print(f"Training complete in {total_mins:.1f} minutes. Model saved to {save_path}")

# ---------------------------
# 8. Single greedy eval with visualization (saves GIF)
# ---------------------------
policy_net.eval()
eval_env = MazeEnv(maze, render_mode="human")
state, info = eval_env.reset()
state_vec = one_hot_state(state, n_states)
frames = []
done = False

for step in range(MAX_STEPS):
    with torch.no_grad():
        q_vals = policy_net(torch.tensor(state_vec))
        action = torch.argmax(q_vals).item()

    state, reward, terminated, truncated, info = eval_env.step(action)
    state_vec = one_hot_state(state, n_states)
    done = terminated or truncated

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

eval_env.close()

if frames:
    gif_path = "metrics/dqn_run.gif"
    imageio.mimsave(gif_path, frames, fps=10, format="GIF")
    print(f"Saved evaluation run animation to {gif_path}")
print(f"Eval finished after {step+1} steps; success={terminated}")
