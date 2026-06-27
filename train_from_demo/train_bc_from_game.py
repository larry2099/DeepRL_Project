import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import cv2

class BCDataset(Dataset):
    def __init__(self, frames, actions, seq_len=4, augment=True):
        self.frames = frames
        self.actions = actions
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return max(0, len(self.actions) - self.seq_len)

    def __getitem__(self, idx):
        seq_frames = self.frames[idx:idx+self.seq_len]
        target = self.actions[idx+self.seq_len-1]

        if self.augment:
            shift_x = random.randint(-3, 3)  # smaller shift
            shift_y = random.randint(-3, 3)
            if shift_x != 0 or shift_y != 0:
                seq_frames = np.roll(seq_frames, shift=shift_x, axis=2)
                seq_frames = np.roll(seq_frames, shift=shift_y, axis=1)
                if shift_x > 0:
                    seq_frames[:, :, :shift_x] = 0
                elif shift_x < 0:
                    seq_frames[:, :, shift_x:] = 0
                if shift_y > 0:
                    seq_frames[:, :shift_y, :] = 0
                elif shift_y < 0:
                    seq_frames[:, shift_y:, :] = 0
            # No brightness noise (to avoid overfitting)

        state = torch.FloatTensor(seq_frames)  # (4,120,120)
        target = torch.LongTensor([target])[0]
        return state, target

# ----- Simpler CNN (3 layers, fewer parameters) -----
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=3):
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
            sample = torch.zeros(1, input_channels, 120, 120)
            n_flatten = self.conv(sample).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features)

def merge_datasets():
    try:
        data1 = np.load("demo_bc.npz")
        frames1 = data1["frames"]
        actions1 = data1["actions"]
        print(f"Loaded recorded demo: {len(frames1)} frames")
    except FileNotFoundError:
        frames1, actions1 = None, None

    try:
        data2 = np.load("video_demo.npz")
        frames2 = data2["frames"]
        actions2 = data2["actions"]
        print(f"Loaded video demo: {len(frames2)} frames")
    except FileNotFoundError:
        frames2, actions2 = None, None

    if frames1 is not None and frames2 is not None:
        frames = np.concatenate([frames1, frames2], axis=0)
        actions = np.concatenate([actions1, actions2], axis=0)
    elif frames1 is not None:
        frames, actions = frames1, actions1
    elif frames2 is not None:
        frames, actions = frames2, actions2
    else:
        raise RuntimeError("No data found.")
    print(f"Total frames after merge: {len(frames)}")
    return frames, actions

def main():
    frames, actions = merge_datasets()

    # Class weights (to handle imbalance)
    unique, counts = np.unique(actions, return_counts=True)
    total = counts.sum()
    class_weights = torch.FloatTensor([total / (len(unique) * c) for c in counts])
    print(f"Class weights: {class_weights}")

    indices = np.random.permutation(len(frames))
    split = int(0.8 * len(frames))
    train_frames, val_frames = frames[indices[:split]], frames[indices[split:]]
    train_actions, val_actions = actions[indices[:split]], actions[indices[split:]]

    train_dataset = BCDataset(train_frames, train_actions, seq_len=4, augment=True)
    val_dataset = BCDataset(val_frames, val_actions, seq_len=4, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(input_channels=4, num_actions=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("Training SIMPLE CNN with 3 layers (less overfitting)...")
    best_val_loss = float('inf')
    for epoch in range(60):
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

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "bc_best.pth")
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/60, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load("bc_best.pth"))
    torch.save(model.state_dict(), "bc_model.pth")
    print("Saved BC model as bc_model.pth")

if __name__ == "__main__":
    main()