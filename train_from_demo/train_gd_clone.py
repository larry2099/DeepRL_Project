from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gd_env import GeometryDashCloneEnv

env = GeometryDashCloneEnv()
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_clone")
print("✅ Training done. Model saved as ppo_clone.zip")