import traceback
from env import GeometryDashEnv

env = GeometryDashEnv()

try:
    while True:
        _, _ = env.reset()
        while True:
            _, _, term, trunc, _ = env.step(1)
            if term or trunc:
                break
        env.restart()
except (Exception, KeyboardInterrupt) as e:
    traceback.print_exception(e)
finally:
    env.close()
