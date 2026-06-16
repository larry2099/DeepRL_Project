"""PPO training script for Geometry Dash."""

import argparse
import signal
import threading

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from callbacks import (
    BestRunRecorder,
    BestRunRecorderCallback,
    PauseCallback,
    RecordBestRunsWrapper,
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
    return parser.parse_args()


def make_sigint_handler(vec_env, recorder):
    def handler(signum, frame):
        _ = signum
        _ = frame

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

    recorder = BestRunRecorder(logdir="./tensorboard/", save_dir="./best_runs/", top_k=10)
    vec_env = RecordBestRunsWrapper(vec_env, recorder)

    signal.signal(signal.SIGINT, make_sigint_handler(vec_env, recorder))

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        device=args.device,
        tensorboard_log="./tensorboard/",
    )

    try:
        callbacks = CallbackList(
            [
                TensorBoardCallback(log_interval=1000),
                BestRunRecorderCallback(recorder, log_interval=10_000, save_interval=50_000),
                PauseCallback(pause_event, stop_event),
            ]
        )
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    except (Exception, KeyboardInterrupt, SystemExit) as e:
        print("ERROR: caught exception", e)
    finally:
        model.save("ppo_geometry_dash")
        vec_env.close()


if __name__ == "__main__":
    main()
