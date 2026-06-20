import traceback
from stable_baselines3 import PPO
from env import GeometryDashEnv
import numpy as np
import torch
from stable_baselines3.common import preprocessing
import sys

import pygame


class Runner:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((8 * 100, 6 * 100))
        self.clock = pygame.time.Clock()
        self.running = True

        self.env = GeometryDashEnv()
        self.model = PPO.load(sys.argv[1], self.env)

        self.frames = []
        self.obs = None
        self.attempt = True

        self.frame = 0

    def run_step(self):
        if not self.running:
            return

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                self.running = False
            if evt.type == pygame.KEYDOWN:
                if evt.dict["key"] == pygame.K_SPACE:
                    self.frame += 1
                if evt.dict["key"] == pygame.K_TAB:
                    self.attempt = True

        self.screen.fill("gray")

        if self.attempt:
            self.run_attempt_step()
        else:
            self.display_frames()

        if self.obs is None and self.attempt:
            self.attempt = False
            self.frame = 0

        pygame.display.flip()
        self.clock.tick(60)

    def debug_frame(self, obs, act):
        h, w, c = obs.shape

        x, _ = self.model.policy.obs_to_tensor(obs)
        x = preprocessing.preprocess_obs(x, self.model.policy.observation_space)
        x.requires_grad_(True)

        self.model.policy.zero_grad()
        _, score, _ = self.model.policy.forward(x)
        score.backward()

        sal = x.grad.abs().cpu().numpy()
        a = np.max(sal)
        b = np.min(sal)
        sal = (sal - b) / (a - b)
        sal *= 255
        sal = sal.astype("uint8")

        obs = obs.astype("uint8")

        for i in range(c):
            arr = np.transpose(obs[:, :, i])
            arr = np.dstack([arr] * 3)
            surf = pygame.surfarray.make_surface(arr)
            self.screen.blit(surf, (10 + (w + 10) * i, 10))

        for i in range(c):
            arr = np.transpose(sal[0, i, :, :])
            arr = np.dstack([arr] * 3)
            surf = pygame.surfarray.make_surface(arr)
            self.screen.blit(surf, (10 + (w + 10) * i, 20 + h))

    def display_frames(self):
        n = len(self.frames)
        if n == 0:
            return

        if self.frame >= n:
            self.frame = 0

        obs, act = self.frames[self.frame]
        self.debug_frame(obs, act)

    def run_attempt_step(self):
        if self.obs is None:
            self.obs, _ = self.env.reset()
            self.frames = []

        action, _ = self.model.predict(self.obs)
        next_obs, _, term, trunc, _ = self.env.step(action)

        if term or trunc:
            self.obs = None
            return

        self.debug_frame(self.obs, action)
        self.frames.append((self.obs, action))
        self.obs = next_obs

    def close(self):
        self.env.close()


runner = Runner()
try:
    while runner.running:
        runner.run_step()

except (Exception, KeyboardInterrupt) as e:
    traceback.print_exception(e)
finally:
    runner.close()
