from enum import Enum
import os
import Box2D
from Box2D import b2Vec2
import math
import numpy as np

import pygame


class Settings:
    SCALE = 30
    RESOLUTION = (800, 600)
    SPEED = 9.6
    GRAVITY = 2 * 0.43 * SPEED * SPEED
    JUMP_VEL = 1.9 * SPEED
    JUMP_PAD_VEL = 2.74 * SPEED
    FPS = 60
    CAM_SPEED = 1e-2
    CAM_SPEED_FAST = 1e-1
    QUERY_SIZE = b2Vec2(0.1, 0.1)
    CAM_FOLLOW = 0

    PLAYER_GRP = 1 << 0
    GROUND_GRP = 1 << 1
    KILL_GRP = 1 << 2
    JUMP_PAD_GRP = 1 << 3
    JUMP_ORB_GRP = 1 << 4
    FINISH_GRP = 1 << 5

    PLAYER_REACT_GRP = GROUND_GRP | KILL_GRP | JUMP_PAD_GRP | JUMP_ORB_GRP | FINISH_GRP

    BLOCK_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.5, 0.5)),
        isSensor=True,
    )
    BLOCK_GROUND = Box2D.b2FixtureDef(
        shape=Box2D.b2EdgeShape(vertex1=(-0.48, 0.5), vertex2=(0.5, 0.5)),
        friction=0,
        filter=Box2D.b2Filter(
            categoryBits=GROUND_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    BLOCK_KILLBOX = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.48, 0.2)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=KILL_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    SPIKE_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(vertices=[(-0.5, -0.5), (0, 0.5), (0.5, -0.5)]),
        isSensor=True,
    )
    SPIKE_KILLBOX = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.2, 0.2)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=KILL_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    JUMP_PAD_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.45, 0.05, (0, -0.45), 0)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=JUMP_PAD_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    JUMP_ORB_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.2, 0.2, (0, 0), math.pi / 4)),
        isSensor=True,
    )
    JUMP_ORB_HITBOX = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.5, 0.5)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=JUMP_ORB_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    SMALL_SPIKE_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(vertices=[(-0.5, -0.5), (0, 0), (0.5, -0.5)]),
        isSensor=True,
    )
    SMALL_SPIKE_KILLBOX = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.2, 0.2, (0, -0.2), 0)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=KILL_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    FINISH_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2CircleShape(radius=0.5, pos=(0, 0)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=FINISH_GRP,
            maskBits=PLAYER_GRP,
        ),
    )

    OBJECT_DATA = [
        {
            "name": "block",
            "shape": [BLOCK_SHAPE, BLOCK_GROUND, BLOCK_KILLBOX],
            "color": 0x0000FF,
        },
        {
            "name": "spike",
            "shape": [SPIKE_SHAPE, SPIKE_KILLBOX],
            "color": 0x880088,
        },
        {
            "name": "jump_pad",
            "shape": [JUMP_PAD_SHAPE],
            "color": 0xFFFF00,
        },
        {
            "name": "jump_orb",
            "shape": [JUMP_ORB_SHAPE, JUMP_ORB_HITBOX],
            "color": 0xFF8800,
        },
        {
            "name": "small_spike",
            "shape": [SMALL_SPIKE_SHAPE, SMALL_SPIKE_KILLBOX],
            "color": 0x880088,
        },
        {
            "name": "finish",
            "shape": [FINISH_SHAPE],
            "color": 0x22FF22,
        },
    ]
    PLAYER_COLOR = 0xFF3333
    GROUND_COLOR = 0x00FF00
    BACKGROUND_COLOR = 0xCCCCCC


class Camera:
    def __init__(self):
        self.scale = Settings.SCALE
        self.world_offset = b2Vec2(-5, -6)
        self.target_offset = b2Vec2(0, 0)
        self.pix_offset = b2Vec2(Settings.RESOLUTION[0] / 2, Settings.RESOLUTION[1] / 2)

    def apply(self, xy: b2Vec2) -> b2Vec2:
        xy = (xy + self.world_offset - self.target_offset) * self.scale
        xy[1] *= -1
        return xy + self.pix_offset

    def applyInv(self, xy: b2Vec2) -> b2Vec2:
        xy -= self.pix_offset
        xy[1] *= -1
        return xy / self.scale - self.world_offset + self.target_offset


class ContactListener(Box2D.b2ContactListener):
    def __init__(self, game: "Game"):
        Box2D.b2ContactListener.__init__(self)
        self.game = game

    def BeginContact(self, contact):
        _, other = self.player_and_other(contact.fixtureA, contact.fixtureB)

        bits = other.filterData.categoryBits
        if bits & Settings.GROUND_GRP != 0:
            self.game.player_on_ground += 1

        if bits & Settings.KILL_GRP != 0:
            self.game.player_dead = True

        if bits & Settings.JUMP_PAD_GRP:
            self.game.player.linearVelocity = b2Vec2(0, Settings.JUMP_PAD_VEL)

        if bits & Settings.JUMP_ORB_GRP:
            self.game.player_in_jump_orb.add(other.body)

        if bits & Settings.FINISH_GRP:
            self.game.player_win = True

    def EndContact(self, contact):
        _, other = self.player_and_other(contact.fixtureA, contact.fixtureB)
        bits = other.filterData.categoryBits

        if bits & Settings.GROUND_GRP != 0:
            self.game.player_on_ground -= 1

        if bits & Settings.JUMP_ORB_GRP:
            orb = other.body
            if orb in self.game.player_in_jump_orb:
                self.game.player_in_jump_orb.remove(orb)

    def player_and_other(self, f1, f2):
        if f1.body == self.game.player:
            return (f1, f2)
        else:
            return (f2, f1)


class Level:
    def __init__(self):
        self.objs = []
        self.freelist = []
        self.start = b2Vec2(0, 0)
        self.world: Box2D.b2World | None = None
        self.bodies = set()

        self.selection = set()

    def deserialize(self, s: str):
        a = s.split("!")
        vals = a[0].split(",")
        self.start.x = float(vals[0])
        self.start.y = float(vals[1])

        s = a[1]
        for obj in s.split(";"):
            a = obj.split(":")
            if len(a) != 2:
                continue

            kind, coords = a[0], a[1]
            b = coords.split(",")
            x, y = b[0], b[1]
            self.objs.append([int(kind), float(x), float(y)])

    def serialize(self):
        s = "{},{}!".format(self.start.x, self.start.y)

        for obj in self.objs:
            if len(obj) == 0:
                continue
            s += "{}:{},{};".format(obj[0], obj[1], obj[2])
        return s

    def build(self, world: Box2D.b2World):
        self.world = world
        self.bodies = set()

        for i, _ in enumerate(self.objs):
            self.create(i)

    def create(self, idx):
        obj = self.objs[idx]
        if len(obj) == 0:
            return
        assert self.world

        o = self.world.CreateKinematicBody(
            position=(obj[1] - self.start.x, obj[2]),
            linearVelocity=(-Settings.SPEED, 0),
            fixtures=Settings.OBJECT_DATA[obj[0]]["shape"],
            userData=idx,
        )

        self.bodies.add(o)

    def place(self, kind, pos: b2Vec2):
        if len(self.freelist) != 0:
            idx = self.freelist.pop()
            self.objs[idx] = [kind, pos.x, pos.y]
        else:
            idx = len(self.objs)
            self.objs.append([kind, pos.x, pos.y])
        self.create(idx)

    def erase(self, pos):
        assert self.world

        class Query(Box2D.b2QueryCallback):
            def __init__(self):
                super().__init__()
                self.obj = None

            def ReportFixture(self, fixture):
                if fixture.body.userData is None:
                    return True
                self.obj = fixture.body
                return False

        query = Query()
        aabb = Box2D.b2AABB(
            lowerBound=pos - Settings.QUERY_SIZE,
            upperBound=pos + Settings.QUERY_SIZE,
        )
        self.world.QueryAABB(query, aabb)

        if query.obj is None:
            return

        idx = query.obj.userData
        if len(self.objs[idx]) != 0:
            self.bodies.remove(query.obj)
            self.freelist.append(idx)
            self.objs[idx] = []
            self.world.DestroyBody(query.obj)

    def select(self, a: b2Vec2, b: b2Vec2):
        assert self.world

        class Query(Box2D.b2QueryCallback):
            def __init__(self):
                super().__init__()
                self.objs = set()

            def ReportFixture(self, fixture):
                if fixture.body.userData is not None:
                    self.objs.add(fixture.body)
                return True

        aabb = Box2D.b2AABB(
            lowerBound=b2Vec2(min(a.x, b.x), min(a.y, b.y)),
            upperBound=b2Vec2(max(a.x, b.x), max(a.y, b.y)),
        )
        query = Query()
        self.world.QueryAABB(query, aabb)
        self.selection = self.selection.union(query.objs)

    def deselect(self):
        self.selection.clear()

    def __iter__(self):
        for obj in self.bodies:
            yield (self.objs[obj.userData][0], obj)


class Mode(Enum):
    NORMAL = 0
    HEADLESS = 1
    NO_RENDER = 2


class Game:
    def __init__(
        self,
        mode: Mode = Mode.NORMAL,
        draw_ray_dirs=None,
        draw_ray_dist=None,
    ):
        if mode == Mode.HEADLESS:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        elif "SDL_VIDEODRIVER" in os.environ:
            os.environ.pop("SDL_VIDEODRIVER")

        self.mode = mode

        if mode != Mode.NO_RENDER:
            pygame.init()
            self.screen = pygame.display.set_mode(Settings.RESOLUTION)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, size=40)
            self.prev_frame_pressed = None
            self.prev_frame_mouse = (False, False, False)
            self.draw_ray_dirs = draw_ray_dirs
            self.draw_ray_dist = draw_ray_dist
            self.rays = []

        self.running = True

        self.editing = False
        self.editor_placed_block: int = 0
        self.editor_select_start = None

        self.dt = 1 / Settings.FPS  # fixed dt for stable env

        self.pixels = None

    def reset(self, level_file=None):
        self.cam = Camera()
        self.running = True

        self.level = Level()
        if level_file is None:
            level_file = self.level_file

        if level_file is None:
            level_file = "lvl1.txt"
            self.level.deserialize("0,0!1:10,0;")
        else:
            with open(level_file, "r") as f:
                self.level.deserialize(f.read())

        self.level_file = level_file

        self.player_on_ground = 0
        self.player_in_jump_orb = set()
        self.player_dead = False
        self.player_win = False

        self.world = Box2D.b2World(
            gravity=(0, -Settings.GRAVITY),
            contactListener=ContactListener(self),
        )

        self.ground = self.world.CreateBody(
            position=(0, -3),
            fixtures=Box2D.b2FixtureDef(
                shape=Box2D.b2PolygonShape(box=(20, 2.5)),
                friction=0,
                filter=Box2D.b2Filter(
                    categoryBits=Settings.GROUND_GRP,
                    maskBits=Settings.PLAYER_GRP,
                ),
            ),
        )

        anchor = self.world.CreateStaticBody(position=(0, 0))
        self.player = self.world.CreateDynamicBody(
            position=(0, self.level.start.y),
            fixedRotation=True,
            fixtures=[
                Box2D.b2FixtureDef(  # shape
                    shape=Box2D.b2PolygonShape(box=(0.5, 0.5)),
                    friction=0,
                    density=1,
                    restitution=0,
                    filter=Box2D.b2Filter(
                        categoryBits=Settings.PLAYER_GRP,
                        maskBits=Settings.PLAYER_REACT_GRP,
                    ),
                ),
            ],
        )
        self.world.CreatePrismaticJoint(
            bodyA=anchor,
            bodyB=self.player,
            localAxisA=b2Vec2(0, 1),
        )

        self.level.build(self.world)

        self.is_jumping = False

    def run(self):
        if not self.running:
            return

        if self.mode != Mode.NO_RENDER:
            self.handleEvents()

        self.update()

        if self.mode != Mode.NO_RENDER:
            self.render()

    def update(self):
        if self.mode != Mode.NO_RENDER and self.keyJustPressed(pygame.K_e):
            self.editing = not self.editing
            self.reset()
        if self.mode != Mode.NO_RENDER and self.keyJustPressed(pygame.K_r):
            self.reset()

        if not self.editing and not self.player_dead and not self.player_win:
            self.updateInGame()
        elif self.editing and self.mode != Mode.NO_RENDER:
            self.updateEditor()

        if self.mode != Mode.NO_RENDER:
            self.prev_frame_pressed = pygame.key.get_pressed()
            self.prev_frame_mouse = pygame.mouse.get_pressed()

    def render(self):
        assert self.mode != Mode.NO_RENDER

        self.pixels = None

        self.screen.fill(Settings.BACKGROUND_COLOR)

        if not self.editing:
            self.drawInGame()
        else:
            self.drawEditor()

        pygame.display.flip()

        if self.mode == Mode.NORMAL:
            self.clock.tick(Settings.FPS)

    def getPixles(self, resolution):
        assert self.mode != Mode.NO_RENDER

        if self.pixels is not None:
            return self.pixels

        surf = pygame.Surface(resolution)
        pygame.transform.scale(self.screen, resolution, surf)
        pixels = pygame.surfarray.array3d(surf).astype(np.float32) / 255.0
        self.pixels = (
            pixels[:, :, 0] * 0.216 + pixels[:, :, 1] * 0.715 + pixels[:, :, 2] * 0.072
        )

        return self.pixels

    def handleEvents(self):
        assert self.mode != Mode.NO_RENDER

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                self.running = False

    def close(self):
        self.running = False
        if self.mode != Mode.NO_RENDER:
            pygame.display.quit()
            pygame.quit()

    def jump(self):
        self.is_jumping = True

    def is_dead(self):
        return self.player_dead

    def is_win(self):
        return self.player_win

    def on_ground(self):
        return self.player_on_ground != 0 or len(self.player_in_jump_orb) != 0

    def raycast(self, direction, max_dist):
        class Query(Box2D.b2RayCastCallback):
            def __init__(self, ground):
                super().__init__()
                self.first = None
                self.dist = 1
                self.ground = ground

            def ReportFixture(self, fixture, _pt, _n, t):
                if fixture.body.userData is None and fixture.body != self.ground:
                    return 1

                if self.first is None or t < self.dist:
                    self.first = fixture.body
                    self.dist = t
                return t

        query = Query(self.ground)
        start: b2Vec2 = self.player.position
        end: b2Vec2 = start + direction * max_dist
        self.world.RayCast(query, start, end)

        if not query.first:
            return (1, -1)

        if query.first == self.ground:
            return (query.dist, 0)

        kind = self.level.objs[query.first.userData][0]
        return (query.dist, kind + 1)

    def drawObject(self, obj, color):
        shape = obj.fixtures[0].shape
        if isinstance(shape, Box2D.b2PolygonShape):
            pts = shape.vertices
            pts = [self.cam.apply(b2Vec2(p) + obj.position) for p in pts]
            pts = [(p.x, p.y) for p in pts]
            pygame.draw.polygon(self.screen, color, pts)
        elif isinstance(shape, Box2D.b2CircleShape):
            center = self.cam.apply(obj.position)
            rad = shape.radius * self.cam.scale
            pygame.draw.circle(self.screen, color, (center.x, center.y), rad)

    def updateInGame(self):
        jump_input = self.is_jumping or (
            self.mode != Mode.NO_RENDER and pygame.key.get_pressed()[pygame.K_SPACE]
        )

        if jump_input and self.on_ground():
            self.player.linearVelocity = (0, Settings.JUMP_VEL)

            if len(self.player_in_jump_orb) != 0:
                orb = self.player_in_jump_orb.pop()
                orb.fixtures[1].filterData.categoryBits &= ~Settings.JUMP_ORB_GRP

        self.world.Step(self.dt, 10, 10)

        self.cam.target_offset += Settings.CAM_FOLLOW * (
            self.player.position - self.cam.target_offset
        )

        self.is_jumping = False

        if self.mode != Mode.NO_RENDER and self.draw_ray_dirs is not None:
            self.rays = [
                self.raycast(d, self.draw_ray_dist) for d in self.draw_ray_dirs
            ]

    def drawInGame(self):
        self.drawObject(self.ground, Settings.GROUND_COLOR)
        self.drawObject(self.player, Settings.PLAYER_COLOR)

        for kind, obj in self.level:
            self.drawObject(obj, Settings.OBJECT_DATA[kind]["color"])

        self.drawRays()

    def drawRays(self):
        if self.draw_ray_dirs is None:
            return
        a = self.cam.apply(self.player.position)
        for (t, k), d in zip(self.rays, self.draw_ray_dirs):
            b = self.player.position + t * d * self.draw_ray_dist
            b = self.cam.apply(b)
            if k == -1:
                col = 0x333333
            elif k == 0:
                col = 0x00FF00
            else:
                col = Settings.OBJECT_DATA[k - 1]["color"]
            pygame.draw.line(self.screen, col, (a.x, a.y), (b.x, b.y))

    def updateEditor(self):
        dt = self.clock.get_time()

        shift = pygame.key.get_mods() & pygame.KMOD_SHIFT != 0
        ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL != 0

        cam_speed = Settings.CAM_SPEED if not shift else Settings.CAM_SPEED_FAST
        pressed = pygame.key.get_pressed()

        if not ctrl:
            if pressed[pygame.K_a]:
                self.cam.world_offset += cam_speed * dt * b2Vec2(1, 0)
            if pressed[pygame.K_d]:
                self.cam.world_offset += cam_speed * dt * b2Vec2(-1, 0)
            if pressed[pygame.K_w]:
                self.cam.world_offset += cam_speed * dt * b2Vec2(0, -1)
            if pressed[pygame.K_s]:
                self.cam.world_offset += cam_speed * dt * b2Vec2(0, 1)
        else:
            if self.keyJustPressed(pygame.K_s):
                with open(self.level_file, "w") as f:
                    f.write(self.level.serialize())

        selection_move_delta = b2Vec2(0, 0)
        if self.keyJustPressed(pygame.K_UP):
            selection_move_delta.y += 1
        if self.keyJustPressed(pygame.K_DOWN):
            selection_move_delta.y -= 1
        if self.keyJustPressed(pygame.K_LEFT):
            selection_move_delta.x -= 1
        if self.keyJustPressed(pygame.K_RIGHT):
            selection_move_delta.x += 1
        if shift:
            selection_move_delta *= 0.1
        for obj in self.level.selection:
            obj.position += selection_move_delta
            self.level.objs[obj.userData][1] = obj.position.x
            self.level.objs[obj.userData][2] = obj.position.y

        (lmb, _, rmb) = self.mouseJustPressed()
        x, y = pygame.mouse.get_pos()

        p_break = self.cam.applyInv(b2Vec2(x, y))
        p_place = b2Vec2(
            math.floor(p_break.x + self.level.start.x + 0.5),
            math.floor(p_break.y + 0.5),
        )

        # accidentally pressing it sometimes messes stuff up for some reason
        # if self.keyJustPressed(pygame.K_z):
        #     self.level.start = p_place

        if lmb and not shift:
            self.level.deselect()
            self.level.place(self.editor_placed_block, p_place)
        elif rmb:
            self.level.deselect()
            self.level.erase(p_break)
        elif lmb and shift:
            self.editor_select_start = p_break

        if self.mouseJustReleased()[0] and self.editor_select_start is not None:
            self.level.select(self.editor_select_start, p_break)
            self.editor_select_start = None

        for i, k in enumerate(range(pygame.K_0, pygame.K_9 + 1)):
            if self.keyJustPressed(k) and len(Settings.OBJECT_DATA) > i:
                self.editor_placed_block = i

    def drawEditor(self):
        ground_level = self.cam.apply(b2Vec2(0, -0.5))[1]
        rect = [
            0,
            ground_level,
            Settings.RESOLUTION[0],
            Settings.RESOLUTION[1] - ground_level,
        ]
        pygame.draw.rect(self.screen, Settings.GROUND_COLOR, rect)

        self.drawObject(self.player, Settings.PLAYER_COLOR)

        start = b2Vec2(0, self.level.start.y)
        pts = [start + (0.2, 0), start + (0, 0.2), start - (0.2, 0), start - (0, 0.2)]
        pts = [self.cam.apply(p) for p in pts]
        pts = [(p.x, p.y) for p in pts]
        pygame.draw.polygon(self.screen, 0x00B827, pts)

        for kind, obj in self.level:
            col = Settings.OBJECT_DATA[kind]["color"]
            if obj in self.level.selection:
                col |= 0x00FF00
            self.drawObject(obj, col)

        self.drawGrid()

        for i, desc in enumerate(Settings.OBJECT_DATA):
            text = self.font.render(
                f"[{i}]{desc['name']}",
                True,
                0xFFFF00 if i != self.editor_placed_block else 0xFF0000,
            )
            self.screen.blit(text, (0, i * 30))

    def drawGrid(self):
        a_world = self.cam.applyInv(b2Vec2(0, 0))
        b_world = self.cam.applyInv(b2Vec2(Settings.RESOLUTION))
        left = math.floor(a_world[0])
        top = math.floor(a_world[1])
        right = math.ceil(b_world[0])
        bot = math.ceil(b_world[1])

        if left > right:
            left, right = right, left
        if bot > top:
            bot, top = top, bot

        left -= 1
        right += 1
        top += 1
        bot -= 1
        for i in range(left, right + 1):
            a = self.cam.apply(b2Vec2(i - 0.5, top))
            b = self.cam.apply(b2Vec2(i - 0.5, bot))
            pygame.draw.line(self.screen, 0, (a.x, a.y), (b.x, b.y), 2)
        for i in range(bot, top + 1):
            a = self.cam.apply(b2Vec2(left, i - 0.5))
            b = self.cam.apply(b2Vec2(right, i - 0.5))
            pygame.draw.line(self.screen, 0, (a.x, a.y), (b.x, b.y), 2)

    def keyJustPressed(self, key):
        if self.prev_frame_pressed is None:
            return pygame.key.get_pressed()[key]
        elif self.prev_frame_pressed[key]:
            return False
        else:
            return pygame.key.get_pressed()[key]

    def mouseJustPressed(self):
        (lmb, mmb, rmb) = pygame.mouse.get_pressed()
        (a, b, c) = self.prev_frame_mouse
        return (lmb and not a, mmb and not b, rmb and not c)

    def mouseJustReleased(self):
        (lmb, mmb, rmb) = pygame.mouse.get_pressed()
        (a, b, c) = self.prev_frame_mouse
        return (not lmb and a, not mmb and b, not rmb and c)


if __name__ == "__main__":
    g = Game()
    g.reset("levels/3.txt")
    while g.running:
        g.run()
