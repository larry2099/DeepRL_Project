import Box2D
import pygame
from Box2D import b2Vec2


class Settings:
    SCALE = 30
    RESOLUTION = (800, 600)
    SPEED = 10
    GRAVITY = 70
    JUMP = 17

    PLAYER_GRP = 0x0001
    GROUND_GRP = 0x0002
    KILL_GRP = 0x0004


class Camera:
    def __init__(self):
        self.scale = Settings.SCALE
        self.world_offset = b2Vec2(0, 0)
        self.pix_offset = b2Vec2(Settings.RESOLUTION[0] / 2, Settings.RESOLUTION[1] / 2)

    def apply(self, xy, world_offset=None):
        xy = (
            xy
            + self.world_offset
            + (b2Vec2() if world_offset is None else world_offset)
        ) * self.scale
        xy[1] *= -1

        return xy + self.pix_offset


class ContactListener(Box2D.b2ContactListener):
    def __init__(self, game):
        Box2D.b2ContactListener.__init__(self)
        self.game = game

    def BeginContact(self, contact):
        player, other = self.player_and_other(contact.fixtureA, contact.fixtureB)

        if other.filterData.categoryBits & Settings.GROUND_GRP != 0:
            self.game.player_on_ground += 1

        if other.filterData.categoryBits & Settings.KILL_GRP != 0:
            self.game.player_dead = True

    def EndContact(self, contact):
        player, other = self.player_and_other(contact.fixtureA, contact.fixtureB)

        if other.filterData.categoryBits & Settings.GROUND_GRP != 0:
            self.game.player_on_ground -= 1

    def player_and_other(self, f1, f2):
        if f1.body == self.game.player:
            return (f1, f2)
        else:
            return (f2, f1)


class Level:
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
            categoryBits=Settings.GROUND_GRP,
            maskBits=Settings.PLAYER_GRP,
        ),
    )
    BLOCK_KILLBOX = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(0.5, 0.2)),
        isSensor=True,
        filter=Box2D.b2Filter(
            categoryBits=Settings.KILL_GRP,
            maskBits=Settings.PLAYER_GRP,
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
            categoryBits=Settings.KILL_GRP,
            maskBits=Settings.PLAYER_GRP,
        ),
    )

    def __init__(self):
        self.objs = [
            10,
            -4.5,
            0,
            11,
            -4.5,
            0,
            12,
            -4.5,
            0,
            20,
            -4.5,
            1,
            21,
            -4.5,
            1,
            30,
            -4.5,
            0,
            33,
            -2.5,
            0,
            36,
            -0.5,
            0,
            39,
            1.5,
            0,
            42,
            3.5,
            0,
        ]
        self.platforms = []
        self.spikes = []

    def build(self, world: Box2D.b2World):
        self.platforms = []
        self.spikes = []

        for i in range(0, len(self.objs), 3):
            if self.objs[i + 2] == Level.BLOCK:
                o = world.CreateKinematicBody(
                    position=(self.objs[i], self.objs[i + 1]),
                    linearVelocity=(-Settings.SPEED, 0),
                    fixtures=[
                        Level.BLOCK_SHAPE,
                        Level.BLOCK_GROUND,
                        Level.BLOCK_KILLBOX,
                    ],
                )
                self.platforms.append(o)
            elif self.objs[i + 2] == Level.SPIKE:
                o = world.CreateKinematicBody(
                    position=(self.objs[i], self.objs[i + 1]),
                    linearVelocity=(-Settings.SPEED, 0),
                    fixtures=[
                        Level.SPIKE_SHAPE,
                        Level.SPIKE_KILLBOX,
                    ],
                )
                self.spikes.append(o)


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.running = True

        self.cam = Camera()
        self.level = Level()
        self.reset()

        self.dt = 1 / 60  # fixed dt for stable env

    def reset(self):
        self.player_on_ground = 0
        self.player_dead = False

        self.world = Box2D.b2World(
            gravity=(0, -Settings.GRAVITY), contactListener=ContactListener(self)
        )
        self.ground = self.world.CreateBody(
            position=(0, -10),
            fixtures=Box2D.b2FixtureDef(
                shape=Box2D.b2PolygonShape(box=(20, 5)),
                friction=0,
                filter=Box2D.b2Filter(
                    categoryBits=Settings.GROUND_GRP,
                    maskBits=Settings.PLAYER_GRP,
                ),
            ),
        )

        anchor = self.world.CreateStaticBody(position=(-5, 0))
        self.player = self.world.CreateDynamicBody(
            position=(-5, 0),
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
        self.world.CreatePrismaticJoint(
            bodyA=anchor,
            bodyB=self.player,
            localAxisA=b2Vec2(0, 1),
        )

        self.level.build(self.world)

    def run(self):
        if not self.running:
            return

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                self.running = False

        if pygame.key.get_pressed()[pygame.K_SPACE] and self.player_on_ground != 0:
            self.player.linearVelocity = (0, Settings.JUMP)

        if pygame.key.get_just_pressed()[pygame.K_r]:
            self.reset()

        self.world.Step(self.dt, 10, 10)
        if self.player_dead:
            self.reset()

        self.screen.fill("gray")

        self.draw_object(self.ground, 0x00FF00)
        self.draw_object(self.player, 0xFF0000)
        for platform in self.level.platforms:
            self.draw_object(platform, 0x0000FF)
        for spike in self.level.spikes:
            self.draw_object(spike, 0xFF00FF)

        pygame.display.flip()
        self.clock.tick(60)

    def draw_object(self, obj, color):
        shape = obj.fixtures[0].shape
        if isinstance(shape, Box2D.b2PolygonShape):
            pts = shape.vertices
            pts = [self.cam.apply(b2Vec2(p), obj.position) for p in pts]
            pygame.draw.polygon(self.screen, color, pts)


if __name__ == "__main__":
    g = Game()
    while g.running:
        g.run()
