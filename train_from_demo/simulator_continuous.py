import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GeometryDashSimulator(gym.Env):
    def __init__(self, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.obstacles = [15,28,42,55,70,85,100,115,130,150,170,190,
                          210,230,260,290,320,350,380,420,460,500]
        self.level_end = 550
        # Continuous action: 0 = no jump, 1 = hold jump for full step
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 + horizon,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance = 0.0
        self.y = 0.0
        self.vy = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        obs_dists = []
        for o in self.obstacles:
            if o > self.distance:
                obs_dists.append(o - self.distance)
                if len(obs_dists) >= self.horizon:
                    break
        while len(obs_dists) < self.horizon:
            obs_dists.append(999.0)
        return np.array([self.y, self.vy] + obs_dists, dtype=np.float32)

    def step(self, action):
        hold_time = float(action[0]) * 0.15
        if hold_time > 0.01 and self.y == 0:
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