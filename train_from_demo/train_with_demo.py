"""
Train a Geometry Dash agent using:
1. Behavior cloning (supervised learning) on a demonstration.
2. Fine‑tuning with PPO on the real game.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from game_env import GeometryDashEnv

# ---------- 1. Load demonstration ----------
print("Loading demonstration...")
data = np.load("demonstration.npz")
frames = data["frames"]      # shape (N, 84, 84)
actions = data["actions"]    # shape (N,)

# Stack frames to create 4‑frame states (the same as env uses)
states = []
target_actions = []
for i in range(len(frames) - 3):
    state = np.stack([frames[i], frames[i+1], frames[i+2], frames[i+3]], axis=0)
    states.append(state)
    # Use the action from the last frame of the stack
    target_actions.append(actions[i+3])

states = np.array(states, dtype=np.float32)
target_actions = np.array(target_actions, dtype=np.int64)
print(f"Loaded {len(states)} state-action pairs.")

# ---------- 2. Behavior cloning: train a CNN to mimic actions ----------
class ImitationCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
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
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(features_dim, 2)  # 2 actions

    def forward(self, observations):
        features = self.cnn(observations)
        features = self.linear(features)
        return features

    def predict_action(self, observations):
        features = self.forward(observations)
        logits = self.policy_head(features)
        return torch.argmax(logits, dim=1)

# Setup dataset and dataloader
dataset = TensorDataset(torch.FloatTensor(states), torch.LongTensor(target_actions))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create a dummy env to get observation space
dummy_env = GeometryDashEnv()
obs_space = dummy_env.observation_space
dummy_env.close()

# Initialize imitation model
imitation_model = ImitationCNN(obs_space, features_dim=256)
optimizer = optim.Adam(imitation_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train for a few epochs
print("\n--- Behavior Cloning (Supervised) ---")
for epoch in range(30):
    total_loss = 0.0
    for batch_states, batch_actions in dataloader:
        optimizer.zero_grad()
        features = imitation_model(batch_states)
        logits = imitation_model.policy_head(features)
        loss = criterion(logits, batch_actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/30, Loss: {avg_loss:.4f}")

# Save the pretrained weights for the policy (the CNN part)
pretrained_state_dict = imitation_model.state_dict()
print("\nBehavior cloning completed. Pretrained weights saved.")

# ---------- 3. Fine‑tune with PPO using the pretrained features ----------
class PretrainedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
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
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        # Load pretrained weights from imitation model
        self.load_state_dict(imitation_model.state_dict(), strict=False)

    def forward(self, observations):
        features = self.cnn(observations)
        return self.linear(features)

# Create environment
env = GeometryDashEnv()
env = DummyVecEnv([lambda: env])

# PPO with pretrained feature extractor
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    verbose=1,
    policy_kwargs={
        "features_extractor_class": PretrainedCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False
    }
)

print("\n--- Fine‑tuning with PPO ---")
model.learn(total_timesteps=500_000)
model.save("ppo_geometry_dash_demo")
print("Training finished. Model saved as ppo_geometry_dash_demo.zip")
env.close()