"""SB3 callbacks for logging Geometry-Dash-specific metrics."""

from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
                metrics["gdash/mean_step_duration_ms"] = float(np.mean(durations)) * 1000
                metrics["gdash/max_step_duration_ms"] = float(np.max(durations)) * 1000

        return metrics
