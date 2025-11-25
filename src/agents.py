import copy
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO as SB3PPO, DQN as SB3DQN

from .mouse_env import MouseMazeEnv, SimpleMazeEnv  # or CleanMazeEnv if you swapped it in
import pickle
from collections import deque


# ============================================================
#  Evolutionary Algorithm
# ============================================================

class SimpleEvolutionaryAgent(nn.Module):
    """
    A simple recurrent MLP policy:
    - Input: observation vector
    - Output: action logits
    Uses a 1-step LSTM to give some memory while staying EA-friendly.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.hx = None
        self.cx = None

    @property
    def _device(self):
        return next(self.parameters()).device

    def reset_state(self):
        """Reset LSTM hidden state at the start of each episode."""
        self.hx = torch.zeros(1, 1, self.hidden_dim, device=self._device)
        self.cx = torch.zeros(1, 1, self.hidden_dim, device=self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept numpy or torch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self._device)
        else:
            x = x.to(self._device)

        if x.ndim == 1:
            x = x.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)

        x = torch.relu(self.embed(x))
        # LSTM expects (batch, seq_len, features); we use seq_len=1
        x = x.unsqueeze(1)

        # If state is None (e.g., someone forgot reset_state), initialize it
        if self.hx is None or self.cx is None:
            self.reset_state()

        out, (self.hx, self.cx) = self.rnn(x, (self.hx, self.cx))
        logits = self.head(out[:, -1, :])
        return logits

    def select_action(self, obs: np.ndarray, temperature: float = 1.0, epsilon: float = 0.1) -> int:
        """
        Stochastic action selection to encourage exploration in EA.
        """
        with torch.no_grad():
            logits = self.forward(obs).squeeze(0)
            num_actions = logits.shape[-1]

            # Epsilon-greedy for diversity
            if torch.rand(1).item() < epsilon:
                return np.random.randint(num_actions)

            if temperature > 1e-6:
                probs = torch.softmax(logits / temperature, dim=-1)
                return torch.multinomial(probs, num_samples=1).item()

            return torch.argmax(logits).item()


class EvolutionaryTrainer:
    """
    Generic evolutionary trainer for any maze env (MouseMazeEnv, SimpleMazeEnv, etc.)
    """

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
        # Allow list of mazes
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

        # Probe env to get obs/action sizes
        probe_env = self.env_cls(self.mazes[0], **self.env_kwargs)
        self.input_dim = probe_env.observation_space.shape[0]
        self.output_dim = probe_env.action_space.n
        self.optimal_path = getattr(probe_env, "optimal_path", None)
        probe_env.close()

        # Initial population
        self.population: List[SimpleEvolutionaryAgent] = [
            self._make_agent() for _ in range(self.pop_size)
        ]

    def _make_agent(self) -> SimpleEvolutionaryAgent:
        agent = SimpleEvolutionaryAgent(self.input_dim, self.output_dim).to(self.device)
        return agent

    def mutate(self, parent: SimpleEvolutionaryAgent) -> SimpleEvolutionaryAgent:
        """
        Gaussian mutation over all weights and biases.
        """
        child = copy.deepcopy(parent).to(self.device)
        for param in child.parameters():
            with torch.no_grad():
                noise = torch.randn_like(param) * self.mutation_power
                param.add_(noise)
        return child

    def evaluate_once(self, agent: SimpleEvolutionaryAgent, temperature: float = 1.0, epsilon: float = 0.1) -> float:
        """
        Run a single episode and return total reward.
        """
        maze = self.mazes[np.random.randint(len(self.mazes))]
        env = self.env_cls(maze, **self.env_kwargs)
        obs, _ = env.reset()
        agent.reset_state()
        total_reward = 0.0
        done = False
        steps = 0
        max_steps = getattr(env, "max_steps", 500)

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
        Average reward over multiple episodes.
        """
        rewards = [self.evaluate_once(agent, temperature, epsilon) for _ in range(self.episodes_per_agent)]
        return float(np.mean(rewards))

    def _save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        best_score: float,
        generation: int,
        filename: str = None,
    ):
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

    def train(self) -> Tuple[SimpleEvolutionaryAgent, Dict[str, List[float]]]:
        """
        Main EA loop.
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

            # Soft exploration schedule
            temp = max(0.3, 1.0 - 0.03 * g)
            eps = max(0.02, 0.12 - 0.006 * g)

            for idx, agent in enumerate(self.population):
                fitness = self.evaluate(agent, temperature=temp, epsilon=eps)
                scores.append(fitness)
                print(f"  Agent {idx:02d} fitness: {fitness:.2f}")

            scores = np.array(scores)
            sort_idx = np.argsort(scores)[::-1]
            gen_best_score = scores[sort_idx[0]]
            gen_mean_score = scores.mean()

            best_scores.append(float(gen_best_score))
            mean_scores.append(float(gen_mean_score))

            print(f"  >> Gen {g+1}: Best = {gen_best_score:.2f}, Mean = {gen_mean_score:.2f}")

            if gen_best_score > global_best_score:
                global_best_score = gen_best_score
                global_best_gen = g + 1
                best_idx = sort_idx[0]
                global_best_state_dict = copy.deepcopy(self.population[best_idx].state_dict())
                self._save_checkpoint(global_best_state_dict, global_best_score, generation=g+1)

            # Selection
            elite_count = max(1, int(self.pop_size * self.elite_frac))
            lucky_count = max(0, int(self.pop_size * self.lucky_frac))

            elite_indices = sort_idx[:elite_count]
            elites = [self.population[i] for i in elite_indices]

            non_elites = sort_idx[elite_count:]
            lucky = []
            if lucky_count > 0 and len(non_elites) > 0:
                lucky_indices = np.random.choice(
                    non_elites,
                    size=min(lucky_count, len(non_elites)),
                    replace=False,
                )
                lucky = [self.population[i] for i in lucky_indices]

            print(f"  Elites: {elite_count}, Lucky survivors: {len(lucky)}")

            parents = elites + lucky
            new_pop: List[SimpleEvolutionaryAgent] = []

            # Copy elites & lucky survivors
            for a in elites + lucky:
                new_pop.append(copy.deepcopy(a).to(self.device))

            # Fill with mutated children
            while len(new_pop) < self.pop_size:
                parent = np.random.choice(parents)
                child = self.mutate(parent)
                new_pop.append(child)

            self.population = new_pop

        self._plot_history(best_scores, mean_scores)

        assert global_best_state_dict is not None, "No best agent state dict saved."
        best_agent = self._make_agent()
        best_agent.load_state_dict(global_best_state_dict)

        print(f"\nTraining complete. Best fitness = {global_best_score:.2f} at generation {global_best_gen}.")
        print(f"Checkpoints and plots are saved in: {self.checkpoint_dir}")

        history = {"best_scores": best_scores, "mean_scores": mean_scores}
        return best_agent, history


# ============================================================
#  Imitation Learning (unchanged)
# ============================================================

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


# ============================================================
#  New: PyTorch DQN (value-based)
# ============================================================

class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idxs))
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgentTorch:
    """
    Vanilla DQN with:
    - Replay buffer
    - Target network
    - Epsilon-greedy exploration
    Suitable for small continuous observation vectors + discrete actions.
    """
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.q_net = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    def act(self, obs: np.ndarray) -> int:
        self.total_steps += 1
        # Epsilon schedule (linear)
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon + frac * (self.epsilon_end - self.epsilon)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def remember(self, s, a, r, ns, done):
        self.replay_buffer.push(s, a, r, ns, done)

    def _update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Q(s,a)
        q_values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            # max_a' Q_target(s', a')
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self._update_target()

        return loss.item()


# ============================================================
#  New: PyTorch PPO (actor-critic, clipped)
# ============================================================

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        super().__init__()
        # Policy
        policy_layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_dims:
            policy_layers.append(nn.Linear(last, h))
            policy_layers.append(nn.ReLU())
            last = h
        self.policy_net = nn.Sequential(*policy_layers)
        self.policy_head = nn.Linear(last, n_actions)

        # Value
        value_layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_dims:
            value_layers.append(nn.Linear(last, h))
            value_layers.append(nn.ReLU())
            last = h
        self.value_net = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(last, 1)

    def forward(self, x):
        raise NotImplementedError

    def get_action_and_value(self, obs_t: torch.Tensor, actions: torch.Tensor | None = None):
        logits = self.policy_head(self.policy_net(obs_t))
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_head(self.value_net(obs_t)).squeeze(-1)
        return actions, log_probs, entropy, value


class PPOAgentTorch:
    """
    Minimal PPO implementation suitable for small discrete-action tasks.
    Uses on-policy rollouts + clipped objective.
    """
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        lr: float = 3e-4,
        batch_size: int = 64,
        update_epochs: int = 10,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.device = torch.device(device)

        self.net = PPOActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def rollout(self, env, steps_per_rollout: int):
        obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []
        obs, _ = env.reset()
        for _ in range(steps_per_rollout):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                actions, log_probs, entropy, value = self.net.get_action_and_value(obs_t)
            action = int(actions.item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(log_probs.item())
            rew_list.append(reward)
            val_list.append(value.item())
            done_list.append(float(done))

            obs = next_obs
            if done:
                obs, _ = env.reset()

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.int64),
            np.array(logp_list, dtype=np.float32),
            np.array(rew_list, dtype=np.float32),
            np.array(val_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
        )

    def compute_gae(self, rewards, values, dones, last_value=0.0):
        """
        Generalized Advantage Estimation.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, old_log_probs, returns, advantages):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logp_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_t)
        idxs = np.arange(n)
        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = adv_t[mb_idx]

                _, logp, entropy, value = self.net.get_action_and_value(mb_obs, mb_actions)
                ratio = torch.exp(logp - mb_old_logp)

                # PPO clipped objective
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = nn.functional.mse_loss(value, mb_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


# ============================================================
#  Stable-Baselines3 factories (optional convenience)
# ============================================================

def get_ppo_model(env):
    """
    Convenience wrapper for SB3 PPO with sensible defaults for small discrete tasks.
    """
    return SB3PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        target_kl=0.01,
    )


def get_dqn_model(env):
    """
    Convenience wrapper for SB3 DQN with standard settings.
    """
    return SB3DQN(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        train_freq=4,
        target_update_interval=1_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
    )


# ============================================================
#  Simple Tabular Q-Learning (debug baseline)
# ============================================================

class QLearningAgent:
    """
    Classic tabular Q-learning with aggressive state discretization.
    Useful to sanity-check that the environment is learnable before
    throwing deep RL at it.
    """
    def __init__(
        self,
        action_size: int,
        lr: float = 0.2,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple[float, ...], np.ndarray] = {}

    def _get_state_key(self, obs: np.ndarray) -> Tuple[float, ...]:
        # Discretize observation to 2 decimal places to bound table size
        return tuple(np.round(obs, 2))

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        state = self._get_state_key(obs)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done: bool):
        state = self._get_state_key(obs)
        next_state = self._get_state_key(next_obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size, dtype=np.float32)

        best_next = np.max(self.q_table[next_state])
        td_target = reward + (0.0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data.get("q_table", {})
        self.epsilon = data.get("epsilon", self.epsilon)


def train_q_learning(env, episodes: int = 500, max_steps: int = 500):
    """
    Clean Q-learning trainer:
    - No smell hacks
    - No stall heuristics
    Just step, update, move on.
    """
    agent = QLearningAgent(action_size=env.action_space.n)
    rewards = []
    successes = []
    steps_per_episode = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        terminated_flag = False
        steps = 0

        for t in range(max_steps):
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            steps += 1
            if done:
                terminated_flag = terminated
                break

        agent.decay_epsilon()
        rewards.append(ep_reward)
        successes.append(1 if terminated_flag else 0)
        steps_per_episode.append(steps)

    return agent, rewards, successes, steps_per_episode
