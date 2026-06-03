import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pyautogui
import cv2
from collections import deque
from utils import DxCamCapture, preprocess_frame
from game_interface import GameInterface

pyautogui.FAILSAFE = False

class GeometryDashEnv(gym.Env):
    def __init__(self, target_size=(84,84), frame_skip=4, max_steps=2000):
        super().__init__()
        self.target_size = target_size
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        # Attach to the already‑open game window
        self.game = GameInterface()
        rect = self.game.get_window_rect()
        if rect:
            region = (rect[0], rect[1], rect[2], rect[3])
        else:
            raise RuntimeError("Could not get window rectangle.")

        self.capture = DxCamCapture(region=region, target_fps=60)
        self.frames = deque(maxlen=4)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(4, target_size[0], target_size[1]),
            dtype=np.float32
        )

        self._activate_window()
        self._start_level()
        self.steps = 0
        self._last_frame = None

    def _activate_window(self):
        self.game._activate_window()
        time.sleep(0.5)

    def _start_level(self):
        for _ in range(4):
            pyautogui.press('space')
            time.sleep(0.2)
        time.sleep(1)

    def _get_obs(self):
        if len(self.frames) == 0:
            return np.zeros((4, self.target_size[0], self.target_size[1]), dtype=np.float32)
        return np.stack(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._activate_window()
        for _ in range(3):
            pyautogui.press('space')
            time.sleep(0.2)
        time.sleep(1)

        self.frames.clear()
        for _ in range(4):
            frame = self.capture.capture_frame()
            if frame is None:
                frame = self._last_frame if self._last_frame is not None else np.zeros((768, 1024, 3), dtype=np.uint8)
            else:
                self._last_frame = frame
            proc = preprocess_frame(frame, self.target_size)
            self.frames.append(proc)

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        if action == 1:
            pyautogui.press('space')

        time.sleep(0.016 * self.frame_skip)

        frame = self.capture.capture_frame()
        if frame is None:
            frame = self._last_frame if self._last_frame is not None else np.zeros((768, 1024, 3), dtype=np.uint8)
        else:
            self._last_frame = frame

        proc = preprocess_frame(frame, self.target_size)
        self.frames.append(proc)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dead = np.mean(gray) < 30

        if dead:
            reward = -10.0
            done = True
        else:
            reward = 0.05
            done = False

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def close(self):
        self.capture.stop()
        self.game.close()