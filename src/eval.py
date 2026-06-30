import math
from Box2D import b2Vec2
from gymnasium.wrappers import FlattenObservation
from env import GameEnv, ObservationKind
import stable_baselines3 as SB3

env = GameEnv(
    obs_kind=ObservationKind.raycasts(count=32, offset=b2Vec2(0.5, 0), spread=math.pi),
    render_mode="human",
)
env = FlattenObservation(env)
model = SB3.PPO.load("ppo.zip", env=env)
obs, _ = env.reset()
while True:
    a, _ = model.predict(obs)
    obs, _, term, _, _ = env.step(a)
    if term:
        obs, _ = env.reset()

