from stable_baselines3 import PPO
from simulator import GeometryDashSimulator

env = GeometryDashSimulator()
model = PPO.load("ppo_sim")

obs, _ = env.reset()
done = False
total_reward = 0
steps = 0

print("Agent playing simulator – watch it complete Level 1")
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    steps += 1
    print(f"Step {steps}: distance={env.distance:.1f}, y={env.y:.2f}, action={action}, reward={reward:.1f}")

print(f"\n🏁 Level complete! Total reward: {total_reward:.2f} in {steps} steps")