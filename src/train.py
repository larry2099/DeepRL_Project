"""PPO training script for Geometry Dash."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_geometry_dash_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--display-base", type=int, default=99)
    parser.add_argument("--stream-port-base", type=int, default=8080)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    env_fns = [
        lambda rank=i: make_geometry_dash_env(
            env_id=rank,
            display_base=args.display_base,
            stream_port_base=args.stream_port_base,
        )
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env.seed(args.seed)

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        device=args.device,
        tensorboard_log="./tensorboard/",
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save("ppo_geometry_dash")

    vec_env.close()


if __name__ == "__main__":
    main()
