import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from gd_clone import Player, ground_exists, obstacles, LEVEL_END, SCROLL_SPEED, GRAVITY, JUMP_SPEED, FLIGHT_THRUST

class GeometryDashCloneEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: y, vy, distance_to_next_obstacle, is_flight (0/1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance = 0.0
        self.player = Player()
        self.player.y = 500 - 20  # GROUND_Y - PLAYER_SIZE
        self.player.vy = 0
        self.player.is_grounded = True
        self.done = False
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # distance to next obstacle
        next_obs = 999.0
        for d, typ, h in obstacles:
            if d > self.distance:
                next_obs = d - self.distance
                break
        is_flight = 1.0 if not ground_exists(self.distance) else 0.0
        return np.array([self.player.y, self.player.vy, next_obs, is_flight], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        hold = float(action[0])

        # Pass distance to player
        self.player.distance = self.distance
        # Update physics (we need a custom update that uses action)
        # We'll manually apply physics here to avoid modifying the clone's update method
        # We'll call the player's update with hold flag
        # But we need to pass hold; we'll modify player.update to accept hold
        # For now, we'll replicate the logic

        # Flight mode
        if not ground_exists(self.distance):
            # Flight: hold gives upward thrust
            if hold > 0.1:
                self.player.vy += FLIGHT_THRUST
            self.player.vy += GRAVITY
            self.player.y += self.player.vy
            self.player.is_grounded = False
        else:
            # Ground mode
            if self.player.is_grounded and hold > 0.5:
                self.player.vy = JUMP_SPEED
                self.player.is_grounded = False
            self.player.vy += GRAVITY
            self.player.y += self.player.vy
            if self.player.y >= 500 - 20:
                self.player.y = 500 - 20
                self.player.vy = 0
                self.player.is_grounded = True

        # Scroll
        self.distance += SCROLL_SPEED

        # Collision
        hit = False
        for d, typ, h in obstacles:
            if abs(d - self.distance) < 2.0:
                if typ == 'spike' and self.player.y >= 500 - 20 - 5:
                    hit = True
                elif typ == 'block' and self.player.y >= 500 - 20 - 15:
                    hit = True
                elif typ == 'double_spike' and self.player.y >= 500 - 20 - 10:
                    hit = True
                elif typ == 'stair' and self.player.y >= 500 - 20 - 10:
                    hit = True
                if hit:
                    break

        if hit:
            reward = -10.0
            self.done = True
        elif self.distance >= LEVEL_END:
            reward = 50.0
            self.done = True
        else:
            reward = 0.05 + (self.distance / LEVEL_END) * 0.05
            self.done = False

        if self.steps > 3000:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        if self.render_mode == "human":
            # We'll reuse the pygame display from clone
            pass