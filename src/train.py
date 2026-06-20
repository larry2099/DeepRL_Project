"""PPO training script for Geometry Dash."""

import argparse
import signal
import threading
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import vgg
import torch
from torch import nn

from callbacks import (
    BestRunRecorder,
    BestRunRecorderCallback,
    PauseCallback,
    RecordBestRunsWrapper,
    RestartCallback,
    TensorBoardCallback,
)
from env import make_geometry_dash_env

pause_event = threading.Event()
stop_event = threading.Event()


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


def make_sigint_handler(vec_env, recorder):
    def handler(signum, frame):
        _ = signum
        _ = frame
        traceback.print_stack()

        while True:
            print(
                "\nCtrl+C menu: "
                "[p]ause, "
                "[r]esume, "
                "[s]tatus, "
                "[k]ey to env, "
                "[h]ard restart env, "
                "[q]uit: ",
                end="",
            )
            choice = input().strip().lower()

            if choice == "p":
                pause_event.set()
                print("paused")
                return
            elif choice == "r":
                pause_event.clear()
                print("resumed")
                return
            elif choice == "s":
                try:
                    lengths = vec_env.get_attr("_episode_length")
                    steps = vec_env.get_attr("_elapsed_steps")
                    best = max(recorder._heap).length if recorder._heap else 0
                    print(f"current lengths: {lengths}")
                    print(f"elapsed steps: {steps}")
                    print(f"best recorded length: {best}")
                except Exception as e:
                    print("status error:", e)
                return
            elif choice == "k":
                print("env index (or 'all'): ", end="")
                idx = input().strip()
                print("key (up/space): ", end="")
                key = input().strip()
                indices = None if idx == "all" else [int(idx)]
                method = "hold_jump" if key == "up" else "interact"
                args = () if method == "hold_jump" else (key, 1.0)
                try:
                    vec_env.env_method(method, *args, indices=indices)
                except Exception as e:
                    print("key error:", e)
                return
            elif choice == "h":
                print("env index (or 'all'): ", end="")
                idx = input().strip()
                indices = None if idx == "all" else [int(idx)]
                try:
                    vec_env.env_method("hard_restart_game", indices=indices)
                except Exception as e:
                    print("restart error:", e)
                return
            elif choice == "q":
                stop_event.set()
                print("stopping...")
                return
            else:
                print("unknown choice")

    return handler


class VggExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, mode="from_scratch"):
        """
        mode = 'from_scratch': train from scratch
               'finetune': load up the default weights, still train them
               'frozen': load up the default weights, do not update them
        """
        features_dim = 512 * 7 * 7
        super().__init__(observation_space, features_dim)
        print("VggExtractor in mode='", mode, "'")

        full_model = vgg.vgg16(
            weights=(vgg.VGG16_Weights.DEFAULT if mode != "from_scratch" else None)
        )

        self.features = full_model.features
        self.avgpool = full_model.avgpool

        if mode == "from_scratch":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        elif mode == "frozen":
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False

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

    signal.signal(signal.SIGINT, make_sigint_handler(vec_env, recorder))

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
                features_extractor_class=VggExtractor,
                features_extractor_kwargs=dict(mode="frozen"),
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
                PauseCallback(pause_event, stop_event),
                CheckpointCallback(
                    save_freq=model.n_steps,
                    save_path="checkpoints",
                    name_prefix="gd",
                    save_replay_buffer=True,
                    save_vecnormalize=True,
                ),
                RestartCallback(
                    interval=model.n_steps // 2,
                ),
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
