from env import GeometryDashEnv

env = GeometryDashEnv()

_, _ = env.reset()
while True:
    _, _, term, trunc, _ = env.step(1)
    if term or trunc:
        break
env.close()
