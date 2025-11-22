import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, DQN
from .mouse_env import MouseMazeEnv
import copy
import pickle

# --- Evolutionary Algorithm ---
class SimpleEvolutionaryAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # Expect x to be numpy array, convert to tensor
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).unsqueeze(0)
        return self.net(x)
    
    def select_action(self, obs):
        with torch.no_grad():
            logits = self.forward(obs)
            return torch.argmax(logits).item()

class EvolutionaryTrainer:
    def __init__(self, maze_data, population_size=20, generations=10):
        self.maze = maze_data
        self.pop_size = population_size
        self.generations = generations
        probe_env = MouseMazeEnv(self.maze)
        self.input_dim = probe_env.observation_space.shape[0]
        self.optimal_path = probe_env.optimal_path
        self.output_dim = 8 # Action space
        
        # Create initial population
        self.population = [SimpleEvolutionaryAgent(self.input_dim, self.output_dim) 
                           for _ in range(population_size)]

    def mutate(self, parent, mutation_power=0.02):
        child = copy.deepcopy(parent)
        for param in child.parameters():
            if len(param.shape) > 0: # Weights/biases
                noise = torch.randn_like(param) * mutation_power
                param.data += noise
        return child

    def evaluate(self, agent):
        env = MouseMazeEnv(self.maze, optimal_path=self.optimal_path)
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 200:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        return total_reward

    def train(self):
        print("Starting Evolutionary Training...")
        for g in range(self.generations):
            scores = []
            for agent in self.population:
                scores.append(self.evaluate(agent))
            
            # Selection: Top 20%
            elite_idx = np.argsort(scores)[-int(self.pop_size*0.2):]
            elites = [self.population[i] for i in elite_idx]
            
            # Reproduction
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                parent = np.random.choice(elites)
                child = self.mutate(parent)
                new_pop.append(child)
            
            self.population = new_pop
            print(f"Generation {g}: Best Reward {max(scores)}")
        
        return self.population[-1] # Return best agent

# --- Imitation Learning (Behavior Cloning) ---
class ImitationAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, obs_batch, act_batch):
        self.optimizer.zero_grad()
        obs = torch.FloatTensor(obs_batch)
        target = torch.LongTensor(act_batch)
        logits = self.net(obs)
        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def predict(self, obs, deterministic=True):
        # Matches SB3 API roughly
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(obs)
            action = torch.argmax(logits).item()
        return action, None

# --- Factories for SB3 ---
def get_ppo_model(env):
    return PPO("MlpPolicy", env, verbose=1)

def get_dqn_model(env):
    return DQN("MlpPolicy", env, verbose=1)

# --- Simple tabular Q-Learning for debugging the environment ---
class QLearningAgent:
    def __init__(self, action_size=8, lr=0.2, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # state_key -> np.ndarray of action values

    def _get_state_key(self, obs):
        # Aggressively discretize continuous observation to keep table size bounded
        return tuple(np.round(obs, 2))

    def act(self, obs, explore=True):
        state = self._get_state_key(obs)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done):
        state = self._get_state_key(obs)
        next_state = self._get_state_key(next_obs)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size, dtype=np.float32)

        best_next = np.max(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data.get("q_table", {})
        self.epsilon = data.get("epsilon", self.epsilon)


def train_q_learning(env, episodes=500, max_steps=500, optimal_steps=1):
    agent = QLearningAgent(action_size=env.action_space.n)
    rewards = []
    successes = []
    step_ratios = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        visited = {}
        best_smell = obs[-1] if len(obs) > 0 else 0.0
        stagnation = 0
        steps = 0
        for _ in range(max_steps):
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            # Penalize staying in loops and encourage smell progress
            state_key = agent._get_state_key(obs)
            visited[state_key] = visited.get(state_key, 0) + 1
            if visited[state_key] > 3:
                reward -= 0.05 * visited[state_key]
            smell = next_obs[-1] if len(next_obs) > 0 else 0.0
            if smell > best_smell + 1e-3:
                best_smell = smell
                stagnation = 0
            else:
                stagnation += 1
            if stagnation > 30:
                truncated = True
                reward -= 1.0
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            steps += 1
            if done:
                break
        agent.decay_epsilon()
        rewards.append(ep_reward)
        successes.append(1 if terminated else 0)
        step_ratios.append(steps / max(1, optimal_steps))
    return agent, rewards, successes, step_ratios
