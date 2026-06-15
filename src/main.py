"""Small example of using the GeometryDash Gymnasium env."""

import logging
import signal

from env import GeometryDashEnv

logging.basicConfig(level=logging.INFO)

env = GeometryDashEnv()
g = env._game


def on_sigint(sig, frame):
    _ = sig
    _ = frame

    global g
    while True:
        s = input("choose action: [i]: interact [j]: jump [r]: release [q]: quit: ")
        if s == "i":
            g.interact()
            return
        elif s == "j":
            g.hold_jump()
            return
        elif s == "r":
            g.release_jump()
            return
        elif s == "q":
            raise Exception("exit")


signal.signal(signal.SIGINT, on_sigint)

try:
    while True:
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = 1000
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            print(f"step {steps}: reward={reward}, dead={info['is_dead']}")

            if done:
                print("episode done")

except Exception as e:
    logging.error(e)
finally:
    env.close()
