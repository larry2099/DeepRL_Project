import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from gymnasium.wrappers import FlattenObservation
import stable_baselines3 as SB3
from env import GameEnv, ObservationKind

print("🎮 Loading trained model...")
model = SB3.PPO.load("ppo")

print("🖥️ Rendering agent. Press Ctrl+C to stop.")
env = GameEnv(obs_kind=ObservationKind.raycasts(), render_mode="human")
env = FlattenObservation(env)

obs, _ = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            obs, _ = env.reset()
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    env.close()