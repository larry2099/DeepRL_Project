"""SB3 callbacks and vec-env wrappers for Geometry Dash."""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper


class TensorBoardCallback(BaseCallback):
    """Read per-env metrics from vectorized envs and write them to TensorBoard."""

    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval != 0:
            return True

        metrics = self._collect_metrics()
        for key, value in metrics.items():
            self.logger.record(key, value)

        return True

    def _collect_metrics(self) -> Dict[str, Any]:
        try:
            episode_rewards = self.training_env.get_attr("_episode_reward")
            episode_lengths = self.training_env.get_attr("_episode_length")
            episode_counts = self.training_env.get_attr("_episode_count")
            death_counts = self.training_env.get_attr("_death_count")
            step_durations = self.training_env.get_attr("_last_step_duration")
            elapsed_steps = self.training_env.get_attr("_elapsed_steps")
            action_counts = self.training_env.get_attr("_action_count")
        except Exception:
            return {}

        metrics = {}
        if episode_rewards:
            metrics["gdash/mean_episode_reward"] = float(np.mean(episode_rewards))
        if episode_lengths:
            metrics["gdash/mean_episode_length"] = float(np.mean(episode_lengths))
        if episode_counts:
            metrics["gdash/total_episodes"] = int(np.sum(episode_counts))
        if death_counts:
            metrics["gdash/total_deaths"] = int(np.sum(death_counts))
        if step_durations:
            durations = [d for d in step_durations if d is not None]
            if durations:
                mean_dur = float(np.mean(durations))
                metrics["gdash/mean_step_duration_ms"] = mean_dur * 1000
                metrics["gdash/max_step_duration_ms"] = float(np.max(durations)) * 1000
                metrics["gdash/mean_env_fps"] = 1.0 / mean_dur if mean_dur > 0 else 0.0
        if elapsed_steps and action_counts:
            total_steps = sum(elapsed_steps)
            total_actions = sum(action_counts)
            if total_steps > 0:
                metrics["gdash/action_rate"] = total_actions / total_steps

        return metrics


@dataclass(order=True)
class Episode:
    """A completed episode ranked by length (survival time)."""

    length: int
    obs: np.ndarray = field(compare=False)
    actions: np.ndarray = field(compare=False)
    rewards: np.ndarray = field(compare=False)


class BestRunRecorder:
    """Track top-K episodes by length and write them to TensorBoard / disk."""

    def __init__(
        self,
        logdir: str = "./tensorboard/",
        save_dir: str = "./best_runs/",
        top_k: int = 10,
    ):
        self.logdir = logdir
        self.save_dir = save_dir
        self.top_k = top_k
        self._heap: List[Episode] = []
        self._total_seen = 0

        os.makedirs(self.save_dir, exist_ok=True)

        # Lazy-import torch to avoid hard dependency at import time.
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.logdir)
        except Exception:
            self._writer = None

    def maybe_add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """Add an episode if it is in the top-K by length."""
        length = len(actions)
        self._total_seen += 1

        if len(self._heap) < self.top_k:
            heapq.heappush(self._heap, Episode(length, obs, actions, rewards))
        elif length > self._heap[0].length:
            heapq.heapreplace(self._heap, Episode(length, obs, actions, rewards))

    def write_to_tensorboard(self, global_step: int) -> None:
        """Log the current best episode as a video in TensorBoard."""
        if self._writer is None or not self._heap:
            return

        best = max(self._heap)
        # obs shape: (T, H, W, C) uint8 -> (1, T, C, H, W) float32 [0,1]
        video = np.expand_dims(best.obs.transpose(0, 3, 1, 2), axis=0).astype(np.float32) / 255.0
        self._writer.add_video("best_run/video", video, global_step, fps=10)

    def save_to_disk(self, global_step: int) -> None:
        """Persist the top-K episodes to disk as .npz files."""
        for rank, ep in enumerate(sorted(self._heap, reverse=True), start=1):
            path = os.path.join(self.save_dir, f"best_run_step{global_step}_rank{rank}_len{ep.length}.npz")
            np.savez(
                path,
                observations=ep.obs,
                actions=ep.actions,
                rewards=ep.rewards,
            )


class RecordBestRunsWrapper(VecEnvWrapper):
    """Capture completed episodes and forward them to a BestRunRecorder."""

    def __init__(self, venv, recorder: BestRunRecorder):
        super().__init__(venv)
        self.recorder = recorder
        self._obs_buffers: Optional[List[List[np.ndarray]]] = None
        self._action_buffers: Optional[List[List[int]]] = None
        self._reward_buffers: Optional[List[List[float]]] = None

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        n = self.num_envs
        self._obs_buffers = [[o] for o in obs]
        self._action_buffers = [[] for _ in range(n)]
        self._reward_buffers = [[] for _ in range(n)]
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        for i, action in enumerate(actions):
            self._action_buffers[i].append(int(action))
        self.venv.step_async(actions)

    def step_wait(self) -> tuple:
        obs, rewards, dones, infos = self.venv.step_wait()

        for i in range(self.num_envs):
            self._obs_buffers[i].append(obs[i])
            self._reward_buffers[i].append(float(rewards[i]))

            if dones[i]:
                ep_obs = np.stack(self._obs_buffers[i][:-1], axis=0)
                ep_actions = np.array(self._action_buffers[i], dtype=np.int64)
                ep_rewards = np.array(self._reward_buffers[i], dtype=np.float32)
                self.recorder.maybe_add(ep_obs, ep_actions, ep_rewards)

                # Start next episode buffer with the auto-reset observation.
                self._obs_buffers[i] = [obs[i]]
                self._action_buffers[i] = []
                self._reward_buffers[i] = []

        return obs, rewards, dones, infos


class BestRunRecorderCallback(BaseCallback):
    """Periodically flush the BestRunRecorder to TensorBoard and disk."""

    def __init__(
        self,
        recorder: BestRunRecorder,
        log_interval: int = 10_000,
        save_interval: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.recorder = recorder
        self.log_interval = log_interval
        self.save_interval = save_interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            self.recorder.write_to_tensorboard(self.num_timesteps)
        if self.num_timesteps % self.save_interval == 0:
            self.recorder.save_to_disk(self.num_timesteps)
        return True
