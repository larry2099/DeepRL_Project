"""OpenAI Gymnasium wrapper for Geometry Dash."""

from __future__ import annotations

import time
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
        frame_stack: int = 4,
        max_episode_steps: int = 2000,
        death_restart_timeout: float = 10.0,
        display: Optional[str] = None,
        stream_port: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.obs_size = obs_size
        self.grayscale = grayscale
        self.frame_stack = frame_stack if grayscale else 1
        self.max_episode_steps = max_episode_steps
        self.death_restart_timeout = death_restart_timeout

        self._game = game.LinuxGame(display=display, stream_port=stream_port)

        self._prev_action = 0
        self._elapsed_steps = 0
        self._action_count = 0

        # Metrics exposed to SB3 callbacks
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_count = 0
        self._death_count = 0
        self._last_step_duration: float | None = None
        self._last_step_start: float | None = None

        # Frame stacking buffer
        h, w = obs_size
        self._frame_buffer = np.zeros(
            (h, w, self.frame_stack), dtype=np.uint8
        )

        self.action_space = spaces.Discrete(2)  # 0 = release/no-op, 1 = hold jump

        channels = self.frame_stack if grayscale else 3
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
        self._action_count = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._last_step_duration = None
        self._frame_buffer.fill(0)

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
        if action == 1:
            self._action_count += 1

        self._last_step_start = time.perf_counter()
        state = self._game.update()
        self._last_step_duration = time.perf_counter() - self._last_step_start
        self._elapsed_steps += 1

        obs = self._get_obs()
        reward = 1.0 if not state.is_dead else -100.0
        reward -= 0 if action == 0 else 0.2
        terminated = state.is_dead
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = self._get_info(state)

        self._episode_reward += reward
        self._episode_length += 1

        self._write_overlay(action, state)

        if terminated:
            self._death_count += 1
            self._episode_count += 1
            self._episode_reward = 0.0
            self._episode_length = 0
        elif truncated:
            self._episode_count += 1
            self._episode_reward = 0.0
            self._episode_length = 0

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._game.last_frame
        return None

    def close(self) -> None:
        self._game.close()

    # ------------------------------------------------------------------
    # Manual control helpers used by the SIGINT menu
    # ------------------------------------------------------------------

    def hold_jump(self) -> None:
        self._game.hold_jump()

    def release_jump(self) -> None:
        self._game.release_jump()

    def interact(self, key: str = "space", delay: float = 1.0) -> None:
        self._game.interact(key=key, delay=delay)

    def hard_restart_game(self) -> None:
        self._game.hard_restart()

    @property
    def best_run_length(self) -> int:
        return self._episode_length

    # ------------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self._game.last_frame is None:
            return self._frame_buffer.copy()

        frame = self._game.last_frame
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(frame, (self.obs_size[1], self.obs_size[0]))
            new_frame = resized[..., np.newaxis]
        else:
            resized = cv2.resize(frame, (self.obs_size[1], self.obs_size[0]))
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Roll buffer and append newest frame.
        self._frame_buffer = np.concatenate(
            [self._frame_buffer[..., 1:], new_frame], axis=-1
        )
        return self._frame_buffer.copy()

    def _get_info(self, state: game.VisionState) -> Dict[str, Any]:
        return {
            "steps": self._elapsed_steps,
            "is_dead": state.is_dead,
        }

    def _write_overlay(self, action: int, state: game.VisionState) -> None:
        try:
            action_str = "JUMP" if action == 1 else "NO-OP"
            dead_str = "DEAD" if state.is_dead else "ALIVE"
            text = (
                f"ACTION: {action_str}\\n"
                f"STEPS: {self._elapsed_steps}\\n"
                f"REWARD: {self._episode_reward:.1f}\\n"
                f"{dead_str}"
            )
            with open(self._game.overlay_path, "w") as f:
                f.write(text)
        except Exception:
            pass

    def _wait_until_alive(self) -> None:
        import time as _time

        deadline = _time.perf_counter() + self.death_restart_timeout
        while _time.perf_counter() < deadline:
            state = self._game.update()
            if not state.is_dead:
                return
        raise RuntimeError("Timed out waiting for level restart after death")


# ------------------------------------------------------------------------------
# Factory for vectorized environments
# ------------------------------------------------------------------------------


def make_geometry_dash_env(
    env_id: int,
    display_base: int = 99,
    stream_port_base: int = 8080,
    **env_kwargs,
) -> GeometryDashEnv:
    """Create a GeometryDashEnv with a unique display and stream port."""
    # SubprocVecEnv workers must not react to the main-process Ctrl+C menu.
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    return GeometryDashEnv(
        display=f":{display_base + env_id}",
        stream_port=stream_port_base + env_id,
        **env_kwargs,
    )
