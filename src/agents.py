import copy
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, DQN

from .mouse_env import MouseMazeEnv, SimpleMazeEnv
import pickle

# --- Evolutionary Algorithm ---
class SimpleEvolutionaryAgent(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) policy:
    - Input: observation vector (e.g., 12-D from MouseMazeEnv)
    - Output: action logits
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.hx = None
        self.cx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept numpy array or torch tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)

        x = torch.relu(self.embed(x))
        # LSTM expects (batch, seq, features); we use seq_len=1
        x = x.unsqueeze(1)
        out, (self.hx, self.cx) = self.rnn(x, (self.hx, self.cx))
        logits = self.head(out[:, -1, :])
        return logits

    def select_action(self, obs: np.ndarray, temperature: float = 1.0, epsilon: float = 0.1) -> int:
        """
        Stochastic action selection to avoid early collapse (helps EA explore).
        """
        with torch.no_grad():
            logits = self.forward(obs).squeeze(0)
            if torch.rand(1).item() < epsilon:
                return np.random.randint(logits.shape[-1])
            if temperature > 1e-6:
                probs = torch.softmax(logits / temperature, dim=-1)
                return torch.multinomial(probs, num_samples=1).item()
            return torch.argmax(logits).item()

    def reset_state(self):
        self.hx = torch.zeros(1, 1, self.hidden_dim)
        self.cx = torch.zeros(1, 1, self.hidden_dim)


class EvolutionaryTrainer:
    def __init__(
        self,
        maze_data: np.ndarray,
        population_size: int = 80,
        generations: int = 50,
        elite_frac: float = 0.1,
        lucky_frac: float = 0.1,
        mutation_power: float = 0.02,
        episodes_per_agent: int = 5,
        checkpoint_dir: str = "checkpoints_ea",
        device: str = "cpu",
        env_cls=MouseMazeEnv,
        env_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Evolutionary Algorithm trainer for the MouseMazeEnv.

        Args:
            maze_data: 2D numpy array for the maze.
            population_size: number of agents in the population.
            generations: number of evolutionary generations.
            elite_frac: fraction of top performers preserved each generation.
            lucky_frac: fraction of randomly chosen non-elite survivors.
            mutation_power: std of Gaussian noise applied to weights.
            episodes_per_agent: how many episodes to average for fitness.
            checkpoint_dir: directory where best agents are saved.
            device: "cpu" or "cuda".
        """
        # Allow a list of mazes for curriculum/multi-episode variety
        self.mazes = maze_data if isinstance(maze_data, (list, tuple)) else [maze_data]
        self.pop_size = population_size
        self.generations = generations
        self.elite_frac = elite_frac
        self.lucky_frac = lucky_frac
        self.mutation_power = mutation_power
        self.episodes_per_agent = episodes_per_agent
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs or {}

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Probe environment to get observation and action sizes + optimal path
        probe_env = self.env_cls(self.mazes[0], **self.env_kwargs)
        self.input_dim = probe_env.observation_space.shape[0]
        self.optimal_path = getattr(probe_env, "optimal_path", None)
        self.output_dim = probe_env.action_space.n

        # Create initial population
        self.population: List[SimpleEvolutionaryAgent] = [
            self._make_agent() for _ in range(self.pop_size)
        ]

    # ---------- Helper methods ----------

    def _make_agent(self) -> SimpleEvolutionaryAgent:
        agent = SimpleEvolutionaryAgent(self.input_dim, self.output_dim).to(self.device)
        return agent

    def mutate(self, parent: SimpleEvolutionaryAgent) -> SimpleEvolutionaryAgent:
        """
        Create a mutated child from a parent by adding Gaussian noise
        to all parameters (weights and biases).
        """
        child = copy.deepcopy(parent)
        for param in child.parameters():
            with torch.no_grad():
                noise = torch.randn_like(param) * self.mutation_power
                param.add_(noise)
        return child

    def evaluate_once(self, agent: SimpleEvolutionaryAgent, temperature: float = 1.0, epsilon: float = 0.1) -> float:
        """
        Run a single episode and return total reward.
        """
        # Sample a maze each episode to encourage generality/curriculum
        maze = self.mazes[np.random.randint(len(self.mazes))]
        env = self.env_cls(maze, **self.env_kwargs)
        obs, _ = env.reset()
        agent.reset_state()
        total_reward = 0.0
        done = False
        steps = 0

        max_steps = env.max_steps

        while not done and steps < max_steps:
            action = agent.select_action(obs, temperature=temperature, epsilon=epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        env.close()
        return total_reward

    def evaluate(self, agent: SimpleEvolutionaryAgent, temperature: float = 1.0, epsilon: float = 0.1) -> float:
        """
        Evaluate an agent over multiple episodes and return its average reward.
        This reduces variance in fitness estimates.
        """
        rewards = []
        for _ in range(self.episodes_per_agent):
            r = self.evaluate_once(agent, temperature=temperature, epsilon=epsilon)
            rewards.append(r)
        return float(np.mean(rewards))

    def _save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        best_score: float,
        generation: int,
        filename: str = None,
    ):
        """
        Save a checkpoint for the best agent.
        """
        if filename is None:
            filename = f"best_agent_gen_{generation:03d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "state_dict": state_dict,
                "best_score": best_score,
                "generation": generation,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            },
            path,
        )
        print(f"  [Checkpoint] Saved best agent of gen {generation} to: {path}")

    def _plot_history(self, best_scores: List[float], mean_scores: List[float]):
        """
        Plot best and mean fitness per generation and save as PNG.
        """
        plt.figure()
        gens = np.arange(1, len(best_scores) + 1)

        plt.plot(gens, best_scores, label="Best Fitness")
        plt.plot(gens, mean_scores, label="Mean Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Total Reward)")
        plt.title("Evolution of Fitness")
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(self.checkpoint_dir, "evolution_curve.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved evolution curve to: {out_path}")

    # ---------- Main training loop ----------

    def train(self) -> Tuple[SimpleEvolutionaryAgent, Dict[str, List[float]]]:
        """
        Run the evolutionary loop.

        Returns:
            best_agent: agent with globally best fitness found.
            history: dict with 'best_scores' and 'mean_scores' lists.
        """
        print("Starting Evolutionary Training...")
        best_scores: List[float] = []
        mean_scores: List[float] = []

        global_best_score = -np.inf
        global_best_state_dict = None
        global_best_gen = -1

        for g in range(self.generations):
            print(f"\n=== Generation {g+1}/{self.generations} ===")
            scores = []

            # Exploratory eval early on; cool over generations
            temp = max(0.3, 1.0 - 0.03 * g)
            eps = max(0.02, 0.12 - 0.006 * g)

            # Evaluate population
            for idx, agent in enumerate(self.population):
                fitness = self.evaluate(agent, temperature=temp, epsilon=eps)
                scores.append(fitness)
                print(f"  Agent {idx:02d} fitness: {fitness:.2f}")

            scores = np.array(scores)
            sorted_idx = np.argsort(scores)[::-1]  # Descending

            gen_best_score = scores[sorted_idx[0]]
            gen_mean_score = scores.mean()

            best_scores.append(float(gen_best_score))
            mean_scores.append(float(gen_mean_score))

            print(f"  >> Gen {g+1}: Best = {gen_best_score:.2f}, Mean = {gen_mean_score:.2f}")

            # Update global best
            if gen_best_score > global_best_score:
                global_best_score = gen_best_score
                global_best_gen = g + 1
                best_agent_idx = sorted_idx[0]
                global_best_state_dict = copy.deepcopy(self.population[best_agent_idx].state_dict())
                self._save_checkpoint(global_best_state_dict, global_best_score, generation=g+1)

            # ----- Selection with elitism + lucky survivors -----

            elite_count = max(1, int(self.pop_size * self.elite_frac))
            lucky_count = max(0, int(self.pop_size * self.lucky_frac))

            elite_indices = sorted_idx[:elite_count]
            elites = [self.population[i] for i in elite_indices]

            # Lucky survivors: randomly choose from non-elites
            non_elite_indices = sorted_idx[elite_count:]
            lucky = []
            if lucky_count > 0 and len(non_elite_indices) > 0:
                lucky_indices = np.random.choice(
                    non_elite_indices,
                    size=min(lucky_count, len(non_elite_indices)),
                    replace=False,
                )
                lucky = [self.population[i] for i in lucky_indices]

            print(f"  Elites: {elite_count}, Lucky survivors: {len(lucky)}")

            parents = elites + lucky  # Used as mutation parents

            # ----- Reproduction / New population -----
            new_population: List[SimpleEvolutionaryAgent] = []

            # Copy elites directly (elitism ensures best solutions survive unchanged)
            for agent in elites:
                new_population.append(copy.deepcopy(agent))

            # Copy lucky survivors directly as well
            for agent in lucky:
                new_population.append(copy.deepcopy(agent))

            # Fill the rest of the population with mutated children
            while len(new_population) < self.pop_size:
                parent = np.random.choice(parents)
                child = self.mutate(parent)
                new_population.append(child)

            self.population = new_population

        # After all generations, make evolution plot
        self._plot_history(best_scores, mean_scores)

        # Build the best agent to return
        assert global_best_state_dict is not None, "No best agent state dict saved."
        best_agent = self._make_agent()
        best_agent.load_state_dict(global_best_state_dict)

        print(f"\nTraining complete. Best fitness = {global_best_score:.2f} at generation {global_best_gen}.")
        print(f"Checkpoints and plots are saved in: {self.checkpoint_dir}")

        history = {
            "best_scores": best_scores,
            "mean_scores": mean_scores,
        }
        return best_agent, history

# --- Imitation Learning (Behavior Cloning) ---
class ImitationAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.hx = None
        self.cx = None

    def reset_state(self):
        self.hx = torch.zeros(1, 1, self.hidden_dim)
        self.cx = torch.zeros(1, 1, self.hidden_dim)

    def forward_logits(self, obs, use_state=False):
        """
        obs: tensor (batch, obs_dim)
        use_state: if True, reuse/ update recurrent state (for online predict).
        """
        x = torch.relu(self.embed(obs))
        x = x.unsqueeze(1)  # (batch, 1, hidden)
        if use_state and self.hx is not None and self.cx is not None:
            out, (self.hx, self.cx) = self.rnn(x, (self.hx, self.cx))
        else:
            out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

    def train_step(self, obs_batch, act_batch):
        self.optimizer.zero_grad()
        obs = torch.FloatTensor(obs_batch)
        target = torch.LongTensor(act_batch)
        logits = self.forward_logits(obs, use_state=False)
        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def predict(self, obs, deterministic=True):
        # Matches SB3 API roughly
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward_logits(obs, use_state=True)
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
