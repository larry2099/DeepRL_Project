"""OpenAI Gymnasium wrapper for Geometry Dash."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import game


class GeometryDashEnv(gym.Env):
    """Gymnasium environment for Geometry Dash.

    Assumptions
    -----------
    * The game window is available at 800x600.
    * A level is already selected/started before ``reset()`` is called.
      The env can auto-retry after death, but it cannot navigate menus.
    * Jump is mapped to the "up" arrow key.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        obs_size: Tuple[int, int] = (120, 160),
        grayscale: bool = True,
        max_episode_steps: int = 2000,
        death_restart_timeout: float = 10.0,
        display: Optional[str] = None,
        stream_port: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.obs_size = obs_size
        self.grayscale = grayscale
        self.max_episode_steps = max_episode_steps
        self.death_restart_timeout = death_restart_timeout

        self._game = game.LinuxGame(display=display, stream_port=stream_port)

        self._prev_action = 0
        self._elapsed_steps = 0

        self.action_space = spaces.Discrete(2)  # 0 = release/no-op, 1 = hold jump

        h, w = obs_size
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, channels),
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._elapsed_steps = 0
        self._prev_action = 0

        if self._game.game_proc is None:
            self._game.open()
            self._game.interact()
            self._game.interact()

        # If the player is dead, the next update() will queue interact.
        # Spin until we are alive again (or timeout).
        state = self._game.last_state
        if state is not None and state.is_dead:
            self._wait_until_alive()

        # One more observation once the level is running.
        state = self._game.update()
        obs = self._get_obs()
        info = self._get_info(state)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        # Emit edge-triggered press/release events.
        if action == 1 and self._prev_action == 0:
            self._game.hold_jump()
        elif action == 0 and self._prev_action == 1:
            self._game.release_jump()
        self._prev_action = action

        state = self._game.update()
        self._elapsed_steps += 1

        obs = self._get_obs()
        reward = 1.0 if not state.is_dead else 0.0
        terminated = state.is_dead
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = self._get_info(state)

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._game.last_frame
        return None

    def close(self) -> None:
        self._game.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self._game.last_frame is None:
            h, w = self.obs_size
            channels = 1 if self.grayscale else 3
            return np.zeros((h, w, channels), dtype=np.uint8)

        frame = self._game.last_frame
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(frame, (self.obs_size[1], self.obs_size[0]))
            return resized[..., np.newaxis]
        else:
            resized = cv2.resize(frame, (self.obs_size[1], self.obs_size[0]))
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _get_info(self, state: game.VisionState) -> Dict[str, Any]:
        return {
            "steps": self._elapsed_steps,
            "is_dead": state.is_dead,
        }

    def _wait_until_alive(self) -> None:
        import time as _time

        deadline = _time.perf_counter() + self.death_restart_timeout
        while _time.perf_counter() < deadline:
            state = self._game.update()
            if not state.is_dead:
                return
        raise RuntimeError("Timed out waiting for level restart after death")
