import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import pyautogui
from collections import deque
from utils import DxCamCapture, preprocess_frame
from game_interface import GameInterface

pyautogui.FAILSAFE = False

class GeometryDashEnv(gym.Env):
    def __init__(self, target_size=(84,84), frame_skip=3, max_steps=2500, stack_frames=4):
        super().__init__()
        self.target_size = target_size
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.stack_frames = stack_frames

        self.game = GameInterface()
        self.game.move_window(100, 100, 1024, 768)
        rect = self.game.get_window_rect()
        if not rect:
            raise RuntimeError("Could not get window rectangle.")
        region = (rect[0], rect[1], rect[2], rect[3])
        self.capture = DxCamCapture(region=region, target_fps=30)

        self.frames = deque(maxlen=stack_frames)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(stack_frames, target_size[0], target_size[1]),
            dtype=np.float32
        )

        self._activate_window()
        self._start_level()
        self.steps = 0
        self._last_frame = None
        self._prev_gray = None
        self.episode_score = 0.0
        self._prev_obstacle_dist = 999.0  # track obstacle distance for clearing detection

    def _activate_window(self):
        self.game._activate_window()
        time.sleep(0.2)

    def _send_jump(self):
        self.game._activate_window()
        pyautogui.press('space')
        time.sleep(0.016)

    def _start_level(self):
        for _ in range(20):
            self._send_jump()
            time.sleep(0.1)
        time.sleep(1)

    def _detect_death(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 30:
            return True
        h, w = gray.shape
        attempt_region = gray[int(h*0.12):int(h*0.30), int(w*0.25):int(w*0.75)]
        if attempt_region.size > 0:
            edges = cv2.Canny(attempt_region, 50, 150)
            if np.sum(edges > 0) / edges.size > 0.03:
                return True
        return False

    def _detect_level_complete(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        complete_region = gray[int(h*0.20):int(h*0.50), int(w*0.20):int(w*0.80)]
        if complete_region.size == 0:
            return False
        if np.mean(complete_region) < 100:
            return False
        edges = cv2.Canny(complete_region, 50, 150)
        if np.sum(edges > 0) / edges.size > 0.03:
            return True
        return False

    def _get_obstacle_distance(self, frame):
        """Return estimated distance to the next obstacle (simulator units)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[int(h*0.7):int(h*0.9), int(w*0.4):int(w*0.9)]
        if roi.size == 0:
            return 999.0
        edges = cv2.Canny(roi, 50, 150)
        # Find first column with significant edge density
        for col in range(edges.shape[1]):
            if np.mean(edges[:, col]) > 100:
                # Map pixel distance to simulator units (approximate)
                dist = (col / roi.shape[1]) * 30 + 5.0
                return dist
        return 999.0

    def _get_obs(self):
        if len(self.frames) == 0:
            return np.zeros((self.stack_frames, self.target_size[0], self.target_size[1]), dtype=np.float32)
        return np.stack(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._activate_window()
        for _ in range(4):
            self._send_jump()
            time.sleep(0.3)
        time.sleep(1)

        self.frames.clear()
        for _ in range(self.stack_frames):
            frame = self.capture.capture_frame()
            if frame is None:
                frame = self._last_frame if self._last_frame is not None else np.zeros((768, 1024, 3), dtype=np.uint8)
            else:
                self._last_frame = frame
            proc = preprocess_frame(frame, self.target_size)
            self.frames.append(proc)

        self.steps = 0
        self.episode_score = 0.0
        self._prev_gray = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY) if self._last_frame is not None else None
        self._prev_obstacle_dist = 999.0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        if action == 1:
            self._send_jump()

        time.sleep(0.016 * self.frame_skip)

        frame = self.capture.capture_frame()
        if frame is None:
            frame = self._last_frame if self._last_frame is not None else np.zeros((768, 1024, 3), dtype=np.uint8)
        else:
            self._last_frame = frame

        proc = preprocess_frame(frame, self.target_size)
        self.frames.append(proc)

        died = self._detect_death(frame)
        completed = self._detect_level_complete(frame)

        # ---- Obstacle distance ----
        current_dist = self._get_obstacle_distance(frame)
        obstacle_cleared = self._prev_obstacle_dist < 20 and current_dist > 50
        self._prev_obstacle_dist = current_dist

        # ---- Progress reward (screen movement) ----
        progress = 0.0
        if self._prev_gray is not None:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_small = cv2.resize(current_gray, (64, 64))
            prev_small = cv2.resize(self._prev_gray, (64, 64))
            diff = np.mean(np.abs(curr_small.astype(float) - prev_small.astype(float)))
            progress = diff * 0.3
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Reward shaping ----
        if died:
            reward = -10.0
            done = True
        elif completed:
            reward = 50.0
            done = True
        else:
            # Base survival
            reward = 0.02

            # Progress reward
            reward += progress

            # Obstacle clearing bonus
            if obstacle_cleared:
                reward += 2.0

            # Timing reward: jump when obstacle distance is in optimal window
            optimal_window = (5.0, 12.0)
            if action == 1:
                if optimal_window[0] < current_dist < optimal_window[1]:
                    reward += 0.3  # good jump
                else:
                    reward -= 0.1  # bad jump

            # Penalty for jumping when no obstacle (far)
            if action == 1 and current_dist > 30:
                reward -= 0.2

            done = False

        if self.steps >= self.max_steps:
            done = True

        self.episode_score += reward
        return self._get_obs(), reward, done, False, {}