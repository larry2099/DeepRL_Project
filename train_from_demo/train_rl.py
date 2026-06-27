import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from game_env import GeometryDashEnv

# ----- Custom CNN (small, fast) -----
class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        features = self.cnn(observations)
        return self.linear(features)

# ----- Reward logger callback -----
class RewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        rewards = self.training_env.get_attr('episode_score')
        if rewards:
            self.episode_rewards.extend(rewards)
            avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            print(f"📊 Avg reward (last 100): {avg:.2f}")

# ----- Main training -----
def main():
    print("="*60)
    print("DIRECT RL TRAINING ON REAL GAME")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start training...")
    input()

    env = GeometryDashEnv()
    env = DummyVecEnv([lambda: env])

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": SmallCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False
        }
    )

    print("\n🚀 Starting training... (This will take several hours)")
    model.learn(total_timesteps=150_000, callback=RewardLogger())
    model.save("ppo_real_game")
    print("\n✅ Training completed. Model saved as ppo_real_game.zip")
    env.close()

if __name__ == "__main__":
    main()