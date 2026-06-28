import os
# Suppress OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import FlattenObservation
import stable_baselines3 as SB3
import traceback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from env import GameEnv, ObservationKind

# ---------- Callback to log episode metrics ----------
class MetricLogger(BaseCallback):
    def __init__(self, log_file="training_metrics.csv", verbose=1):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_count = 0
        # Write header
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length", "loss", "timestep"])
        print(f"📊 CSV log file created: {os.path.abspath(log_file)}")

    def _on_step(self):
        env = self.training_env.envs[0]
        if hasattr(env, 'episode_returns') and len(env.episode_returns) > 0:
            self.episode_count += 1
            reward = env.episode_returns.pop(0)
            length = env.episode_lengths.pop(0)
            loss = self.model.logger.name_to_value.get('train/loss', 0.0)
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_count,
                    reward,
                    length,
                    loss,
                    self.num_timesteps
                ])
            if self.episode_count % 100 == 0:
                print(f"📝 Logged episode {self.episode_count}: reward={reward:.1f}, length={length}")
        return True

# ---------- Main training ----------
print("🚀 Starting training...")

env = GameEnv(obs_kind=ObservationKind.raycasts())
env = Monitor(env)
env = FlattenObservation(env)

model = SB3.PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
)

callback = MetricLogger()

try:
    model.learn(total_timesteps=1_000_000, callback=callback)
except (KeyboardInterrupt, Exception) as e:
    print("Training interrupted:")
    traceback.print_exc()
finally:
    model.save("ppo")
    print("✅ Model saved as ppo.zip")

# ---------- Generate plot from CSV ----------
csv_path = "training_metrics.csv"
if os.path.exists(csv_path):
    print(f"📊 Found CSV at {os.path.abspath(csv_path)}. Generating plot...")
    try:
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:  # skip empty lines
                    data.append([float(x) for x in row])
        if not data:
            print("⚠️ CSV file is empty. No plot generated.")
        else:
            data = np.array(data)
            episodes = data[:, 0]
            rewards = data[:, 1]
            lengths = data[:, 2]
            losses = data[:, 3]

            window = 50
            def smooth(y):
                if len(y) < window:
                    return y
                return np.convolve(y, np.ones(window)/window, mode='valid')

            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
            if len(episodes) >= window:
                axes[0].plot(episodes[window-1:], smooth(rewards), color='blue', linewidth=2, label='Smoothed')
            axes[0].set_ylabel('Episode Reward')
            axes[0].set_title('Training Reward')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
            if len(episodes) >= window:
                axes[1].plot(episodes[window-1:], smooth(lengths), color='green', linewidth=2, label='Smoothed')
            axes[1].set_ylabel('Episode Length')
            axes[1].set_title('Episode Length')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            axes[2].plot(episodes, losses, alpha=0.5, color='red', label='Loss')
            if len(episodes) >= window:
                axes[2].plot(episodes[window-1:], smooth(losses), color='red', linewidth=2, label='Smoothed')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Training Loss')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig("training_summary.png", dpi=150)
            print(f"✅ Plot saved as {os.path.abspath('training_summary.png')}")
    except Exception as e:
        print(f"❌ Error generating plot: {e}")
else:
    print(f"⚠️ No CSV found at {os.path.abspath(csv_path)}. Plot not generated.")

print("🎉 Training script finished.")