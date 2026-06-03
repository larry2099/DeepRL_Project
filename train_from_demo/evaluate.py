from stable_baselines3 import PPO
from game_env import GeometryDashEnv

env = GeometryDashEnv()
model = PPO.load("ppo_geometry_dash")

obs, _ = env.reset()
total_reward = 0
done = False

print("Agent is playing. Watch the game window. Press Ctrl+C to stop.")
try:
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    print(f"Total reward: {total_reward:.2f}")
    env.close()