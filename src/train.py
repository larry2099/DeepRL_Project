"""PPO training script for Geometry Dash."""

import argparse
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import alexnet
import torch

from callbacks import (
    BestRunRecorder,
    BestRunRecorderCallback,
    RecordBestRunsWrapper,
    RestartCallback,
    TensorBoardCallback,
)
from env import make_geometry_dash_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--display-base", type=int, default=99)
    parser.add_argument("--stream-port-base", type=int, default=8080)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        features_dim = 256 * 6 * 6
        super().__init__(observation_space, features_dim)
        full_model = alexnet()

        self.features = full_model.features
        self.avgpool = full_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def main():
    args = parse_args()

    env_fns = [
        lambda rank=i: Monitor(
            make_geometry_dash_env(
                env_id=rank,
                display_base=args.display_base,
                stream_port_base=args.stream_port_base,
            )
        )
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env.seed(args.seed)

    recorder = BestRunRecorder(
        logdir="./tensorboard/", save_dir="./best_runs/", top_k=10
    )
    vec_env = RecordBestRunsWrapper(vec_env, recorder)

    if args.checkpoint is None:
        classifier = [4096, 4096]
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            device=args.device,
            tensorboard_log="./tensorboard/",
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=classifier, vf=classifier),
                features_extractor_class=Extractor,
                share_features_extractor=True,
            ),
        )
    else:
        model = PPO.load(args.checkpoint, vec_env)

    try:
        callbacks = CallbackList(
            [
                TensorBoardCallback(log_interval=model.n_steps),
                BestRunRecorderCallback(
                    recorder,
                    log_interval=model.n_steps,
                    save_interval=model.n_steps,
                ),
                CheckpointCallback(
                    save_freq=model.n_steps,
                    save_path="checkpoints",
                    name_prefix="gd",
                    save_replay_buffer=True,
                    save_vecnormalize=True,
                ),
                RestartCallback(interval=model.n_steps // 2),
            ]
        )
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    except (Exception, KeyboardInterrupt, SystemExit) as e:
        traceback.print_exception(e)
        print("ERROR: caught exception", e)
    finally:
        model.save("ppo_geometry_dash")
        vec_env.close()


if __name__ == "__main__":
    main()
