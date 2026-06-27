import time
import numpy as np
import torch
from game_env import GeometryDashEnv
from dqn_agent import DQNAgent

def train_dqn():
    print("="*60)
    print("TRAINING DUELING DQN – IMPROVED REWARD")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start training...")
    input()

    env = GeometryDashEnv(target_size=(84,84), frame_skip=3, max_steps=2500)
    state_shape = (4, 84, 84)
    num_actions = env.action_space.n

    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.997,
        buffer_size=30000,
        batch_size=64,
        target_update=500
    )

    episode = 0
    step_count = 0
    total_steps = 80000  # ~4-5 hours at 6 FPS

    print("Starting training...\n")
    best_reward = -float('inf')

    while step_count < total_steps:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done and episode_steps < env.max_steps:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            step_count += 1

            loss = agent.update()
            if step_count % 500 == 0:
                print(f"Step {step_count}/{total_steps} | Epsilon: {agent.epsilon:.3f} | Loss: {loss:.4f}")

            if step_count >= total_steps:
                break

        agent.decay_epsilon()
        print(f"Episode {episode+1} | Steps: {episode_steps} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("dqn_best.pth")
            print(f"   New best model saved! Reward: {episode_reward:.2f}")

        episode += 1

    agent.save("dqn_final.pth")
    print("Training completed. Model saved as dqn_final.pth")
    env.close()

if __name__ == "__main__":
    train_dqn()