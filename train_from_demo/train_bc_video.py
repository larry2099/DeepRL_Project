import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

class BCDataset(Dataset):
    def __init__(self, frames, actions, augment=True):
        self.frames = frames
        self.actions = actions
        self.augment = augment

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        state = self.frames[idx].copy()
        action = self.actions[idx]
        if self.augment:
            # small shifts for robustness
            shift_x = random.randint(-4, 4)
            shift_y = random.randint(-4, 4)
            if shift_x != 0 or shift_y != 0:
                state = np.roll(state, shift=shift_x, axis=1)
                state = np.roll(state, shift=shift_y, axis=0)
                if shift_x > 0:
                    state[:, :shift_x] = 0
                elif shift_x < 0:
                    state[:, shift_x:] = 0
                if shift_y > 0:
                    state[:shift_y, :] = 0
                elif shift_y < 0:
                    state[shift_y:, :] = 0
            # brightness noise
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.02, state.shape)
                state = np.clip(state + noise, 0, 1)
        state = torch.FloatTensor(state).unsqueeze(0)  # (1,84,84)
        action = torch.LongTensor([action])[0]
        return state, action

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_actions=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, 84, 84)
            n_flatten = self.conv(sample).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features)

def main():
    data = np.load("video_demo.npz")
    frames = data["frames"]
    actions = data["actions"]

    # Split train/val
    indices = np.random.permutation(len(frames))
    split = int(0.8 * len(frames))
    train_frames, val_frames = frames[indices[:split]], frames[indices[split:]]
    train_actions, val_actions = actions[indices[:split]], actions[indices[split:]]

    train_dataset = BCDataset(train_frames, train_actions, augment=True)
    val_dataset = BCDataset(val_frames, val_actions, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training BC on video pseudo-labels...")
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for batch_states, batch_targets in train_loader:
            batch_states, batch_targets = batch_states.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_targets in val_loader:
                batch_states, batch_targets = batch_states.to(device), batch_targets.to(device)
                logits = model(batch_states)
                loss = criterion(logits, batch_targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/30, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the CNN weights (will be loaded into DQN's feature extractor)
    torch.save(model.state_dict(), "bc_from_video.pth")
    print("Saved BC model as bc_from_video.pth")

if __name__ == "__main__":
    main()