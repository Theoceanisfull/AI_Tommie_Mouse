import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

from src.maze_generator import MazeGenerator
from src.mouse_env import MouseMazeEnv
from src.agents import (
    get_ppo_model,
    get_dqn_model,
    EvolutionaryTrainer,
    ImitationAgent,
    train_q_learning,
    QLearningAgent,
)

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_random_maze(complexity="Easy", split="train"):
    """Loads a random maze of specific complexity from dataset"""
    meta_path = os.path.join(DATA_DIR, "maze_metadata.csv")
    if not os.path.exists(meta_path):
        print("No dataset found. Run generation first.")
        return None
    
    df = pd.read_csv(meta_path)
    filtered = df[df["difficulty"] == complexity]
    if "split" in filtered.columns and split:
        filtered = filtered[filtered["split"] == split]
    if filtered.empty:
        return None
    
    row = filtered.sample(1).iloc[0]
    maze = np.load(os.path.join(DATA_DIR, row["filename"]))
    path = None
    if "path_file" in row and isinstance(row["path_file"], str):
        path_path = os.path.join(DATA_DIR, row["path_file"])
        if os.path.exists(path_path):
            path = np.load(path_path)
    return maze, row["optimal_steps"], path


def _direction_to_action(prev, curr):
    dr = curr[0] - prev[0]
    dc = curr[1] - prev[1]
    mapping = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    return mapping.get((dr, dc), None)


def build_imitation_dataset(limit=200, split="train"):
    """Collect (obs, action) pairs by walking optimal paths."""
    meta_path = os.path.join(DATA_DIR, "maze_metadata.csv")
    if not os.path.exists(meta_path):
        print("No dataset found. Run generation first.")
        return None, None

    df = pd.read_csv(meta_path)
    if "split" in df.columns:
        df = df[df["split"] == split]
    if df.empty:
        print("No mazes available for imitation dataset.")
        return None, None

    # Subsample for quick training
    df = df.sample(min(limit, len(df)), random_state=0)

    observations = []
    actions = []

    for _, row in df.iterrows():
        maze = np.load(os.path.join(DATA_DIR, row["filename"]))
        path = None
        if "path_file" in row and isinstance(row["path_file"], str):
            path_path = os.path.join(DATA_DIR, row["path_file"])
            if os.path.exists(path_path):
                path = np.load(path_path)

        env = MouseMazeEnv(maze, optimal_path=path)
        path_seq = path.tolist() if path is not None else env._compute_optimal_path()[0]

        if path_seq is None or len(path_seq) < 2:
            continue

        obs, _ = env.reset()
        for i in range(1, len(path_seq)):
            prev = tuple(path_seq[i - 1])
            curr = tuple(path_seq[i])
            action = _direction_to_action(prev, curr)
            if action is None:
                continue
            observations.append(obs)
            obs, _, _, _, _ = env.step(action)
            actions.append(action)

    if not observations:
        print("No imitation data collected.")
        return None, None

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def plot_training_curves(output_path, rewards=None, success_curve=None, step_ratio_curve=None, losses=None, title="Training progress"):
    rows = 0
    if rewards is not None:
        rows += 1
    if success_curve is not None or step_ratio_curve is not None:
        rows += 1
    if losses is not None:
        rows += 1
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]
    idx = 0
    if rewards is not None:
        axes[idx].plot(rewards, label="Episode reward")
        axes[idx].set_ylabel("Reward")
        axes[idx].legend()
        idx += 1
    if success_curve is not None or step_ratio_curve is not None:
        if success_curve is not None:
            axes[idx].plot(success_curve, label="Cumulative success rate")
        if step_ratio_curve is not None:
            axes[idx].plot(step_ratio_curve, label="Cumulative step ratio")
        axes[idx].set_ylabel("Performance")
        axes[idx].legend()
        idx += 1
    if losses is not None:
        axes[idx].plot(losses, label="Loss")
        axes[idx].set_ylabel("Loss")
        axes[idx].legend()
    axes[-1].set_xlabel("Episodes / Epochs")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def evaluate_policy(algo, model, env, optimal_steps, episodes=10):
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        steps = 0
        done = False
        success = False
        while not done and steps < env.max_steps:
            if algo in ["ppo", "dqn", "imitation"]:
                action, _ = model.predict(obs, deterministic=True)
            elif algo == "ea":
                action = model.select_action(obs)
            elif algo == "qlearn":
                action = model.act(obs, explore=False)
            else:
                action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                success = True
        step_ratio = steps / max(1, optimal_steps)
        results.append((success, step_ratio))
    if not results:
        return 0.0, np.inf
    successes = [int(s[0]) for s in results]
    ratios = [s[1] for s in results]
    return float(np.mean(successes)), float(np.mean(ratios))

def train(algo, episodes=1000):
    print(f"Training {algo}...")
    
    # For simplicity in this demo, we train on ONE medium maze to prove concept.
    # In a real full run, you would wrap the env to cycle through mazes.
    maze_bundle = load_random_maze("Medium", split="train")
    if maze_bundle is None:
        return
    maze, optimal, path = maze_bundle
    eval_bundle = load_random_maze("Medium", split="test") or maze_bundle

    env = MouseMazeEnv(maze, optimal_path=path)
    env = Monitor(env)

    if algo == "ppo":
        model = get_ppo_model(env)
        model.learn(total_timesteps=episodes * 20) # approx steps
        model.save(f"{MODEL_DIR}/ppo_mouse")
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = MouseMazeEnv(eval_maze, optimal_path=eval_path)
        success_rate, avg_ratio = evaluate_policy("ppo", model, eval_env, eval_optimal)
        print(f"PPO eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")
    
    elif algo == "dqn":
        model = get_dqn_model(env)
        model.learn(total_timesteps=episodes * 20)
        model.save(f"{MODEL_DIR}/dqn_mouse")
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = MouseMazeEnv(eval_maze, optimal_path=eval_path)
        success_rate, avg_ratio = evaluate_policy("dqn", model, eval_env, eval_optimal)
        print(f"DQN eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")
        
    elif algo == "ea":
        trainer = EvolutionaryTrainer(maze, generations=20)
        best_agent = trainer.train()
        torch.save(best_agent.state_dict(), f"{MODEL_DIR}/ea_mouse.pth")
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = MouseMazeEnv(eval_maze, optimal_path=eval_path)
        success_rate, avg_ratio = evaluate_policy("ea", best_agent, eval_env, eval_optimal)
        print(f"EA eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")

    elif algo == "qlearn":
        agent, rewards, successes, step_ratios = train_q_learning(env, episodes=episodes, max_steps=env.max_steps, optimal_steps=optimal)
        np.save(os.path.join(MODEL_DIR, "qlearn_rewards.npy"), np.array(rewards, dtype=np.float32))
        agent.save(os.path.join(MODEL_DIR, "qlearn_mouse.pkl"))
        success_curve = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        step_curve = np.cumsum(step_ratios) / np.arange(1, len(step_ratios) + 1)
        plot_training_curves(
            os.path.join(MODEL_DIR, "qlearn_training.png"),
            rewards=rewards,
            success_curve=success_curve,
            step_ratio_curve=step_curve,
            title="Q-Learning Training"
        )
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = MouseMazeEnv(eval_maze, optimal_path=eval_path)
        success_rate, avg_ratio = evaluate_policy("qlearn", agent, eval_env, eval_optimal)
        print(f"Q-learning eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")
        
    elif algo == "imitation":
        print("Building expert trajectories for imitation...")
        obs_data, act_data = build_imitation_dataset(limit=300, split="train")
        if obs_data is None:
            print("Imitation training aborted: no expert data.")
            return
        obs_dim = int(np.prod(env.observation_space.shape))
        agent = ImitationAgent(obs_dim, env.action_space.n)
        batch_size = 64
        epochs = max(5, episodes // 10)
        epoch_losses = []
        for epoch in range(epochs):
            perm = np.random.permutation(len(obs_data))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(obs_data), batch_size):
                idx = perm[start:start+batch_size]
                loss = agent.train_step(obs_data[idx], act_data[idx])
                epoch_loss += loss
                batches += 1
            avg_loss = epoch_loss / max(1, batches)
            epoch_losses.append(avg_loss)
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")
        torch.save(agent.state_dict(), f"{MODEL_DIR}/imitation_mouse.pth")
        plot_training_curves(
            os.path.join(MODEL_DIR, "imitation_training.png"),
            losses=epoch_losses,
            title="Imitation Training Loss"
        )
        eval_maze, eval_optimal, eval_path = eval_bundle
        eval_env = MouseMazeEnv(eval_maze, optimal_path=eval_path)
        success_rate, avg_ratio = evaluate_policy("imitation", agent, eval_env, eval_optimal)
        print(f"Imitation eval success rate: {success_rate:.2f}, step ratio: {avg_ratio:.2f}")

    print(f"{algo} training complete.")

def visualize(algo, difficulty):
    maze_bundle = load_random_maze(difficulty, split="test")
    if maze_bundle is None:
        print("No maze available to visualize. Generate data first.")
        return
    maze, optimal_steps, path = maze_bundle
    env = MouseMazeEnv(maze, optimal_path=path, render_mode="human")
    
    model = None
    
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(f"{MODEL_DIR}/ppo_mouse")
        elif algo == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(f"{MODEL_DIR}/dqn_mouse")
        elif algo == "ea":
            from src.agents import SimpleEvolutionaryAgent
            h, w = maze.shape
            model = SimpleEvolutionaryAgent(h*w, 8)
            model.load_state_dict(torch.load(f"{MODEL_DIR}/ea_mouse.pth"))
        elif algo == "qlearn":
            model = QLearningAgent(action_size=8)
            model.load(os.path.join(MODEL_DIR, "qlearn_mouse.pkl"))
        elif algo == "imitation":
            obs_dim = int(np.prod(env.observation_space.shape))
            model = ImitationAgent(obs_dim, env.action_space.n)
            model.load_state_dict(torch.load(f"{MODEL_DIR}/imitation_mouse.pth"))
    except FileNotFoundError:
        print(f"Model {algo} not found. Train it first.")
        return

    print(f"Visualizing {algo} on {difficulty} maze. Optimal Steps: {optimal_steps}")
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        if algo in ["ppo", "dqn"]:
            action, _ = model.predict(obs, deterministic=True)
        elif algo == "ea":
            action = model.select_action(obs)
        elif algo == "qlearn":
            action = model.act(obs, explore=False)
        elif algo == "imitation":
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Check events
        if terminated:
            print("Mouse found the cheese!")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train", "visualize"], required=True)
    parser.add_argument("--algo", choices=["ppo", "dqn", "ea", "imitation", "qlearn"], default="ppo")
    parser.add_argument("--difficulty", choices=["Easy", "Medium", "Hard"], default="Medium")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        gen = MazeGenerator(DATA_DIR)
        gen.generate_dataset()
    elif args.mode == "train":
        train(args.algo)
    elif args.mode == "visualize":
        visualize(args.algo, args.difficulty)
