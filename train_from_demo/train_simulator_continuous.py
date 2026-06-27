from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from simulator_continuous import GeometryDashSimulator

env = GeometryDashSimulator(horizon=5)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_sim_continuous")
print("✅ Training done. Model saved as ppo_sim_continuous.zip")