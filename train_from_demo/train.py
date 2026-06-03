from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
import torch.nn as nn
from game_env import GeometryDashEnv

# Custom CNN feature extractor (deeper than the default NatureCNN)
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 4 (stacked frames)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute the shape by doing a forward pass with a dummy tensor
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        features = self.cnn(observations)
        return self.linear(features)

# Environment
env = GeometryDashEnv()
env = DummyVecEnv([lambda: env])

# Stop training when the mean reward exceeds 50 (agent can complete level)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    callback_after_eval=callback_on_best
)

# PPO with custom CNN policy and tuned hyperparameters
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=1e-4,          # lower learning rate for stable learning
    n_steps=4096,                # collect more steps per update
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,               # higher entropy to encourage exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    policy_kwargs={
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": False
    }
)

print("Starting training...")
model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save("ppo_geometry_dash")
print("Training finished. Model saved as ppo_geometry_dash.zip")
env.close()