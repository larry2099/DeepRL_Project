import Box2D
import pygame
from Box2D import b2Vec2
import math


class Settings:
    SCALE = 30
    RESOLUTION = (800, 600)
    SPEED = 10
    GRAVITY = 70
    JUMP = 17

    PLAYER_GRP = 0x0001
    GROUND_GRP = 0x0002
    KILL_GRP = 0x0004

    CAM_SPEED = 1e-2
    QUERY_SIZE = b2Vec2(0.1, 0.1)

    BLOCK = 0
    SPIKE = 1

    BLOCK_SHAPE = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.5, 0.5)),
        isSensor=True,
    )
    BLOCK_GROUND = Box2D.b2FixtureDef(
        shape=Box2D.b2EdgeShape(vertex1=(-0.5, 0.5), vertex2=(0.5, 0.5)),
        friction=0,
        filter=Box2D.b2Filter(
            categoryBits=GROUND_GRP,
            maskBits=PLAYER_GRP,
        ),
    )
    BLOCK_KILLBOX = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.5, 0.2)),
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

    OBJECT_DATA = [
        {"shape": [BLOCK_SHAPE, BLOCK_GROUND, BLOCK_KILLBOX], "name": "block"},
        {"shape": [SPIKE_SHAPE, SPIKE_KILLBOX], "name": "spike"},
    ]


class Camera:
    def __init__(self):
        self.scale = Settings.SCALE
        self.world_offset = b2Vec2(-5, -6)
        self.pix_offset = b2Vec2(Settings.RESOLUTION[0] / 2, Settings.RESOLUTION[1] / 2)

    def apply(self, xy: b2Vec2) -> b2Vec2:
        xy = (xy + self.world_offset) * self.scale
        xy[1] *= -1
        return xy + self.pix_offset

    def applyInv(self, xy: b2Vec2) -> b2Vec2:
        xy -= self.pix_offset
        xy[1] *= -1
        return xy / self.scale - self.world_offset


class ContactListener(Box2D.b2ContactListener):
    def __init__(self, game):
        Box2D.b2ContactListener.__init__(self)
        self.game = game

    def BeginContact(self, contact):
        _, other = self.player_and_other(contact.fixtureA, contact.fixtureB)

        if other.filterData.categoryBits & Settings.GROUND_GRP != 0:
            self.game.player_on_ground += 1

        if other.filterData.categoryBits & Settings.KILL_GRP != 0:
            self.game.player_dead = True

    def EndContact(self, contact):
        _, other = self.player_and_other(contact.fixtureA, contact.fixtureB)

        if other.filterData.categoryBits & Settings.GROUND_GRP != 0:
            self.game.player_on_ground -= 1

    def player_and_other(self, f1, f2):
        if f1.body == self.game.player:
            return (f1, f2)
        else:
            return (f2, f1)


class Level:
    def __init__(self):
        self.objs = [
            [0, 10, 0],
            [0, 13, 0],
            [0, 13, 1],
            [0, 13, 2],
            [1, 20, 0],
            [1, 21, 0],
        ]
        self.freelist = []

        self.world: Box2D.b2World | None = None
        self.platforms = set()
        self.spikes = set()

    def deserialize(self, s):
        pass

    def serialize(self):
        pass

    def build(self, world: Box2D.b2World):
        self.world = world
        self.platforms = set()
        self.spikes = set()

        for i, _ in enumerate(self.objs):
            self.create(i)

    def create(self, idx):
        obj = self.objs[idx]
        if len(obj) == 0:
            return
        assert self.world

        o = self.world.CreateKinematicBody(
            position=(obj[1], obj[2]),
            linearVelocity=(-Settings.SPEED, 0),
            fixtures=Settings.OBJECT_DATA[obj[0]]["shape"],
            userData=idx,
        )

        if obj[0] == 0:
            self.platforms.add(o)
        elif obj[0] == 1:
            self.spikes.add(o)

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
                if self.obj is not None:
                    return False
                self.obj = fixture.body
                return True

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
            if self.objs[idx][0] == 0:
                self.platforms.remove(query.obj)
            elif self.objs[idx][0] == 1:
                self.spikes.remove(query.obj)
            else:
                raise Exception("invalid object", idx, self.objs[idx])

            self.freelist.append(idx)
            self.objs[idx] = []
            self.world.DestroyBody(query.obj)


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(size=40)
        self.running = True

        self.level = Level()
        self.reset()

        self.editing = False
        self.editor_selected: int = 0

        self.dt = 1 / 60  # fixed dt for stable env

    def reset(self):
        self.cam = Camera()

        self.player_on_ground = 0
        self.player_dead = False

        self.world = Box2D.b2World(
            gravity=(0, -Settings.GRAVITY), contactListener=ContactListener(self)
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

        self.player = self.world.CreateDynamicBody(
            position=(0, 0),
            fixedRotation=True,
            fixtures=[
                Box2D.b2FixtureDef(  # shape
                    shape=Box2D.b2PolygonShape(box=(0.5, 0.5)),
                    friction=0,
                    density=1,
                    restitution=0,
                    filter=Box2D.b2Filter(
                        categoryBits=Settings.PLAYER_GRP,
                        maskBits=Settings.GROUND_GRP | Settings.KILL_GRP,
                    ),
                ),
            ],
        )

        self.level.build(self.world)

    def run(self):
        if not self.running:
            return

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                self.running = False

        if pygame.key.get_just_pressed()[pygame.K_e]:
            self.editing = not self.editing
            self.reset()

        if not self.editing:
            self.updateInGame()
        else:
            self.updateEditor()

        self.screen.fill("gray")

        if not self.editing:
            self.drawInGame()
        else:
            self.drawEditor()

        pygame.display.flip()
        self.clock.tick(60)

    def draw_object(self, obj, color):
        shape = obj.fixtures[0].shape
        if isinstance(shape, Box2D.b2PolygonShape):
            pts = shape.vertices
            pts = [self.cam.apply(b2Vec2(p) + obj.position) for p in pts]
            pygame.draw.polygon(self.screen, color, pts)

    def updateInGame(self):
        if pygame.key.get_just_pressed()[pygame.K_r]:
            self.reset()
        if pygame.key.get_pressed()[pygame.K_SPACE] and self.player_on_ground != 0:
            self.player.linearVelocity = (0, Settings.JUMP)
        self.world.Step(self.dt, 10, 10)

        if self.player_dead:
            self.reset()

    def drawInGame(self):
        self.draw_object(self.ground, 0x00FF00)
        self.draw_object(self.player, 0xFF0000)

        for platform in self.level.platforms:
            self.draw_object(platform, 0x0000FF)
        for spike in self.level.spikes:
            self.draw_object(spike, 0xFF00FF)

    def updateEditor(self):
        dt = self.clock.get_time()

        if pygame.key.get_pressed()[pygame.K_a]:
            self.cam.world_offset += Settings.CAM_SPEED * dt * b2Vec2(1, 0)
        if pygame.key.get_pressed()[pygame.K_d]:
            self.cam.world_offset += Settings.CAM_SPEED * dt * b2Vec2(-1, 0)
        if pygame.key.get_pressed()[pygame.K_w]:
            self.cam.world_offset += Settings.CAM_SPEED * dt * b2Vec2(0, -1)
        if pygame.key.get_pressed()[pygame.K_s]:
            self.cam.world_offset += Settings.CAM_SPEED * dt * b2Vec2(0, 1)

        lmb, _, rmb, _, _ = pygame.mouse.get_just_pressed()
        x, y = pygame.mouse.get_pos()
        p = self.cam.applyInv(b2Vec2(x, y))

        if lmb:
            p = b2Vec2(math.floor(p.x + 0.5), math.floor(p.y + 0.5))
            self.level.place(self.editor_selected, p)
        elif rmb:
            self.level.erase(p)

        for i, k in enumerate(range(pygame.K_0, pygame.K_9 + 1)):
            if pygame.key.get_just_pressed()[k] and len(Settings.OBJECT_DATA) > i:
                self.editor_selected = i

    def drawEditor(self):
        ground_level = self.cam.apply(b2Vec2(0, -0.5))[1]
        rect = [
            0,
            ground_level,
            Settings.RESOLUTION[0],
            Settings.RESOLUTION[1] - ground_level,
        ]
        pygame.draw.rect(self.screen, 0x00FF00, rect)

        self.draw_object(self.player, 0xFF0000)

        for platform in self.level.platforms:
            self.draw_object(platform, 0x0000FF)
        for spike in self.level.spikes:
            self.draw_object(spike, 0xFF00FF)

        self.drawGrid()

        for i, desc in enumerate(Settings.OBJECT_DATA):
            text = self.font.render(
                f"[{i}]{desc['name']}",
                True,
                0xFFFF00 if i != self.editor_selected else 0xFF0000,
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
            pygame.draw.line(self.screen, 0, a, b, 2)
        for i in range(bot, top + 1):
            a = self.cam.apply(b2Vec2(left, i - 0.5))
            b = self.cam.apply(b2Vec2(right, i - 0.5))
            pygame.draw.line(self.screen, 0, a, b, 2)


if __name__ == "__main__":
    g = Game()
    while g.running:
        g.run()
