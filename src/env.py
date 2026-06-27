import gymnasium as gym

import os
import time

from game import Game, Settings


class ObservationKind:
    PIXELS = 0
    RAYCASTS = 1

    def __init__(self, **kwargs):
        self.kind = None
        self.resolution = None
        self.max_sight = None

        self.include_on_ground = False
        if "include_on_ground" in kwargs:
            self.include_on_ground = kwargs["include_on_ground"]

    @staticmethod
    def pixels(resolution=(160, 120), **kwargs):
        s = ObservationKind(**kwargs)
        s.kind = ObservationKind.PIXELS
        s.resolution = resolution
        return s

    @staticmethod
    def raycasts(count=16, max_sight=10, **kwargs):
        s = ObservationKind(**kwargs)
        s.kind = ObservationKind.RAYCASTS
        s.resolution = count

    def space(self) -> gym.spaces.Dict:
        if self.kind == ObservationKind.PIXELS:
            d = {"pixels": gym.spaces.Box(0, 1, self.resolution)}
        elif self.kind == ObservationKind.RAYCASTS:
            obj_count = len(Settings.OBJECT_DATA)
            d = {
                "dists": gym.spaces.Box(0, 1, self.resolution),
                "hits": gym.spaces.MultiDiscrete(
                    [obj_count + 1 for _ in range(self.resolution)]
                ),
            }

        if self.include_on_ground:
            d["on_ground"] = gym.spaces.Discrete(2)

        return gym.spaces.Dict(d)

    def observe(self, game: Game):
        if self.kind == ObservationKind.PIXELS:
            pixels = game.getPixles(self.resolution)
            d = {"pixels": pixels}
            if self.include_on_ground:
                d["on_ground"] = game.on_ground()

            return d
        else:
            raise Exception("todo: implement observation {}".format(self.kind))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": Settings.FPS}

    def __init__(
        self, render_mode=None, obs_kind=ObservationKind.pixels(), levels_dir="levels"
    ):
        super().__init__()

        self.observation_space = obs_kind.space()
        self.action_space = gym.spaces.Discrete(2)

        self.obs_kind = obs_kind
        self.render_mode = render_mode
        self.levels_dir = levels_dir
        # TODO: probably better to render the game at observed resolution straight away
        self.game = Game(headless=self.render_mode is None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "level" in options:
            picked_level = os.path.join(self.levels_dir, options["level"])
        else:
            levels = os.listdir(self.levels_dir)
            self.np_random.shuffle(levels)
            picked_level = os.path.join(self.levels_dir, levels[0])

        self.game.reset(picked_level)
        self.game.run()

        return self.obs_kind.observe(self.game), {}

    def step(self, action):
        if action == 1:
            self.game.jump()
        self.game.run()

        obs = self.obs_kind.observe(self.game)
        reward = 1.0 if not self.game.is_dead() else 0.0
        return obs, reward, self.game.is_dead() or not self.game.running, False, {}

    def close(self):
        self.game.close()


if __name__ == "__main__":
    env = GameEnv()

    while True:
        obs, _ = env.reset()
        total_rew = 0
        steps = 0
        start = time.perf_counter()

        while True:
            act = env.action_space.sample()
            obs, rew, term, _, _ = env.step(act)
            steps += 1
            total_rew += rew

            if term:
                break

        duration = time.perf_counter() - start
        print(f"died!, reward={total_rew}, fps={steps / duration:.2f}")
