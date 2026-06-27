from gymnasium.wrappers import FlattenObservation
import stable_baselines3 as SB3
import traceback

from env import GameEnv, ObservationKind

env = GameEnv(obs_kind=ObservationKind.raycasts())
env = FlattenObservation(env)

model = SB3.PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
)
try:
    model.learn(total_timesteps=1_000_000)
except (KeyboardInterrupt, Exception) as e:
    traceback.print_exception(e)
finally:
    model.save("ppo")


del model
model = SB3.PPO.load("ppo")

env = GameEnv(obs_kind=ObservationKind.raycasts(), render_mode="human")
env = FlattenObservation(env)
obs, _ = env.reset()
while True:
    a, _ = model.predict(obs)
    obs, _, term, _, _ = env.step(a)
    if term:
        obs, _ = env.reset()
