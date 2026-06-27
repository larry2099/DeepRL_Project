import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from simulator_corrected import GeometryDashSimulator

def run_and_visualise():
    env = GeometryDashSimulator()
    model = PPO.load("ppo_sim_corrected")

    obs, _ = env.reset()
    done = False
    steps = 0
    distances = []
    heights = []
    actions = []
    rewards = []
    obstacle_positions = env.obstacles

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        distances.append(env.distance)
        heights.append(env.y)
        actions.append(action)
        rewards.append(reward)
        steps += 1

    print(f"Level complete! Steps: {steps}, Total reward: {sum(rewards):.2f}")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Height over distance
    axes[0].plot(distances, heights, label='Player height (y)', color='blue')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Height')
    axes[0].set_title('Player height over distance')
    axes[0].grid(True, alpha=0.3)

    # Mark obstacles (vertical lines)
    for obs in obstacle_positions:
        axes[0].axvline(x=obs, color='red', linestyle=':', alpha=0.5)
    axes[0].legend()

    # 2. Action over distance (0=no jump, 1=jump)
    axes[1].plot(distances, actions, label='Action', color='green', drawstyle='steps-post')
    axes[1].set_ylabel('Action (0=no jump, 1=jump)')
    axes[1].set_title('Actions over distance')
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Reward over distance
    axes[2].plot(distances, rewards, label='Reward', color='purple')
    axes[2].set_xlabel('Distance')
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Reward over distance')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('simulator_results.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    run_and_visualise()