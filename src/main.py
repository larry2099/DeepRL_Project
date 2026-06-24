import time
import traceback
from env import GeometryDashEnv
import random

env = GeometryDashEnv()

try:
    while True:
        _, _ = env.reset()
        t0 = time.perf_counter()
        frame_cnt = 0

        act = 0
        while True:
            if random.random() > 0.1:
                act = 1 - act
            _, _, term, trunc, _ = env.step(act)
            frame_cnt += 1
            if term or trunc:
                break

        t1 = time.perf_counter()
        fps = frame_cnt / (t1 - t0)
        print(f"fps={fps:.2f}")

        env.restart()
except (Exception, KeyboardInterrupt) as e:
    traceback.print_exception(e)
finally:
    env.close()
