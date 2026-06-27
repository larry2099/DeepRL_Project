import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GeometryDashSimulator(gym.Env):
    def __init__(self):
        super().__init__()
        # Stereo Madness obstacle layout (distance, type, height)
        self.obstacles = [
            (15, 'spike', 1.0),
            (28, 'spike', 1.0),
            (42, 'spike', 1.0),
            (55, 'block', 1.0),
            (70, 'spike', 1.0),
            (85, 'double_spike', 1.0),
            (100, 'block', 1.0),
            (115, 'spike', 1.0),
            (130, 'spike', 1.0),
            (150, 'stair', 1.0),
            (170, 'block', 1.0),
            (190, 'spike', 1.0),
            (210, 'spike', 1.0),
            (230, 'block', 1.0),
            (260, 'spike', 1.0),
            (290, 'spike', 1.0),
            (320, 'block', 1.0),
            (350, 'spike', 1.0),
            (380, 'double_spike', 1.0),
            (420, 'block', 1.0),
            (460, 'spike', 1.0),
            (500, 'spike', 1.0)
        ]
        self.level_end = 550

        # ----- FIXED PHYSICS -----
        self.gravity = -0.8          # negative = pulls down
        self.jump_force = 10.0       # positive = pushes up
        self.scroll_speed = 5.0
        # --------------------------

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance = 0.0
        self.y = 0.0          # height above ground
        self.vy = 0.0         # vertical velocity (positive = up)
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # distance to the next obstacle
        next_obs = 999.0
        for dist, typ, h in self.obstacles:
            if dist > self.distance:
                next_obs = dist - self.distance
                break
        return np.array([self.y, self.vy, next_obs], dtype=np.float32)

    def step(self, action):
        # Jump only if on ground
        if action == 1 and self.y == 0:
            self.vy = self.jump_force

        # Physics
        self.vy += self.gravity   # gravity pulls down
        self.y += self.vy

        # Ground collision
        if self.y < 0:
            self.y = 0
            self.vy = 0

        # Scroll forward
        self.distance += self.scroll_speed

        # Collision detection
        hit = False
        for dist, typ, height in self.obstacles:
            if abs(dist - self.distance) < 2.0:
                if typ == 'spike':
                    if self.y < 0.3:
                        hit = True
                elif typ == 'block':
                    if self.y < 1.0:
                        hit = True
                elif typ == 'double_spike':
                    if self.y < 0.5:
                        hit = True
                elif typ == 'stair':
                    if self.y < 0.5:
                        hit = True
                if hit:
                    break

        if hit:
            reward = -10.0
            self.done = True
        elif self.distance >= self.level_end:
            reward = 50.0
            self.done = True
        else:
            reward = 0.1 + (self.distance / self.level_end) * 0.05
            self.done = False

        return self._get_obs(), reward, self.done, False, {}