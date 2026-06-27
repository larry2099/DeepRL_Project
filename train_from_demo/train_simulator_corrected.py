from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from simulator_corrected import GeometryDashSimulator

env = GeometryDashSimulator()
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_sim_corrected")
print("✅ Training done. Model saved as ppo_sim_corrected.zip")