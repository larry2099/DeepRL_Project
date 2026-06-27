import math
from Box2D import b2Vec2
import gymnasium as gym
import numpy as np

import os
import time

from game import Game, Settings, Mode


class ObservationKind:
    PIXELS = 0
    RAYCASTS = 1

    def __init__(self, **kwargs):
        self.kind = None
        self.resolution = None
        self.max_sight = None
        self.directions = None

        self.frame_count: int = 1
        self.include_on_ground = False
        self.frames = {}

        if "include_on_ground" in kwargs:
            self.include_on_ground = kwargs["include_on_ground"]
        if "frame_count" in kwargs:
            self.frame_count = kwargs["frame_count"]
        self.counter = 0

    @staticmethod
    def pixels(resolution=(160, 120), **kwargs):
        s = ObservationKind(**kwargs)
        s.kind = ObservationKind.PIXELS
        s.resolution = resolution
        s.frames = s.space().sample((s.frame_count, None))
        return s

    @staticmethod
    def raycasts(count=16, spread=math.pi / 2, max_sight=10, **kwargs):
        s = ObservationKind(**kwargs)
        s.kind = ObservationKind.RAYCASTS
        s.resolution = count
        s.max_sight = max_sight
        s.directions = []

        dt = spread / count
        for i in range(count):
            t = dt * (i - count // 2)
            s.directions.append(b2Vec2(math.cos(t), math.sin(t)))

        s.frames = s.space().sample((s.frame_count, None))

        return s

    def space(self):
        if self.kind == ObservationKind.PIXELS:
            d = {"pixels": gym.spaces.Box(0, 1, self.resolution)}
        elif self.kind == ObservationKind.RAYCASTS:
            obj_count = len(Settings.OBJECT_DATA)

            d = {
                "dists": gym.spaces.Box(0, 1, (self.resolution,)),
                "kinds": gym.spaces.MultiDiscrete(
                    [obj_count + 2 for _ in range(self.resolution)],
                    start=[-1 for _ in range(self.resolution)],
                ),
            }

        if self.include_on_ground:
            d["on_ground"] = gym.spaces.Discrete(2)

        return gym.spaces.Sequence(gym.spaces.Dict(d), stack=True)

    def observe(self, game: Game):
        i = self.counter % self.frame_count
        self.counter += 1

        if self.kind == ObservationKind.PIXELS:
            pixels = game.getPixles(self.resolution)
            self.frames["pixels"][i] = pixels
        else:
            dists = []
            kinds = []

            for direction in self.directions:
                (dist, kind) = game.raycast(direction, self.max_sight)
                dists.append(dist)
                kinds.append(kind)

            self.frames["dists"][i] = np.array(dists, dtype=np.float32)
            self.frames["kinds"][i] = np.array(kinds, dtype=np.int64)

        if self.include_on_ground:
            self.frames["on_ground"][i] = game.on_ground()

        return self.frames


class GameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": Settings.FPS}

    def __init__(
        self,
        render_mode=None,
        obs_kind=ObservationKind.pixels(),
        levels_dir="levels",
    ):
        super().__init__()

        self.observation_space = obs_kind.space()
        self.action_space = gym.spaces.Discrete(2)

        self.obs_kind = obs_kind
        self.render_mode = render_mode
        self.levels_dir = levels_dir

        mode: Mode = Mode.NORMAL

        if self.render_mode is None:
            mode = Mode.HEADLESS

        if self.obs_kind.kind == ObservationKind.RAYCASTS and self.render_mode is None:
            mode = Mode.NO_RENDER

        if self.obs_kind.kind == ObservationKind.RAYCASTS:
            self.game = Game(
                mode=mode,
                draw_ray_dirs=self.obs_kind.directions,
                draw_ray_dist=self.obs_kind.max_sight,
            )
        else:
            self.game = Game(mode=mode)

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
        # TODO: tweak this?
        reward = 1.0 if not self.game.is_dead() else 0.0
        term = self.game.is_dead() or not self.game.running or self.game.is_win()

        return obs, reward, term, False, {}

    def close(self):
        self.game.close()


""" 
Env usage

You can configure the env to collect 2 different kinds of observations:
    - ObservationKind.pixels: default, 2d grayscale image, resolution 
    can be changed using the 'resolution' param of the constructor.
    Runs at 100fps on my machine.

    - ObservationKind.raycast: 'count' rays get shot out of the player,
    in an angle of 'spread' (i.e. if spread=pi/2, the total view angle is 90deg)
    they record the first object they hit (-1 = no hit, 0 = ground, 
    1.. = object types such as block, spike, ...) and normalized distance.
    Runs at 1500fps on my machine.

Both kinds also accept optional parameters:
    - include_on_ground: (default=False) add an on_ground field to 
    the observations i.e. is the player on ground, or able to use a 
    jump orb.
    
    - frame_count: (default=1) how many frames worth of observations 
    to keep. For example we can keep 4 previous frames of pixels, or 
    4 previous raycast results

You can set render_mode to "human" (default is None), then there will 
be a pygame window. In human mode the game is locked to 60fps.

GameEnv.reset() by default picks a random level every time, you can pass an 
optional parameter 'level' to pick a specific level.

You can tweak the reward function in GameEnv.step.

"""

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    from PIL import Image

    obs, _ = env.reset(options={"level": "2.txt"})

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

    Image.fromarray(obs["pixels"][0] * 255.0).show()

    duration = time.perf_counter() - start
    print(f"died!, reward={total_rew}, fps={steps / duration:.2f}")
