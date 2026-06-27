import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GeometryDashSimulator(gym.Env):
    def __init__(self):
        super().__init__()
        self.obstacles = [
            15, 28, 42, 55, 70, 85, 100, 115, 130, 150, 170, 190,
            210, 230, 260, 290, 320, 350, 380, 420, 460, 500
        ]
        self.level_end = 550
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance = 0.0
        self.y = 0.0
        self.vy = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        next_obs = min([o - self.distance for o in self.obstacles if o > self.distance], default=999)
        return np.array([self.y, self.vy, next_obs], dtype=np.float32)

    def step(self, action):
        if action == 1 and self.y == 0:
            self.vy = -10.0
        self.vy += 0.8
        self.y += self.vy
        if self.y < 0:
            self.y = 0
            self.vy = 0
        self.distance += 5.0
        hit = any(abs(o - self.distance) < 2.0 and self.y < 0.5 for o in self.obstacles)
        if hit:
            return self._get_obs(), -10.0, True, False, {}
        elif self.distance >= self.level_end:
            return self._get_obs(), 50.0, True, False, {}
        else:
            return self._get_obs(), 0.1 + (self.distance / self.level_end) * 0.05, False, False, {}