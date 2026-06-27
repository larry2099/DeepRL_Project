import pygame
import sys

# Window
WIDTH, HEIGHT = 800, 600
GROUND_Y = 500
SCALE = 3.0          # pixels per unit – makes obstacles more visible

# Player
PLAYER_SIZE = 20
PLAYER_X = 150       # fixed x position

# Physics (adjusted)
GRAVITY = 0.8
JUMP_SPEED = -10.0
FLIGHT_THRUST = -7.0
SCROLL_SPEED = 4.0   # slower scroll for better visibility

# Level layout (distance, type)
obstacles = [
    (15, 'spike'),
    (28, 'spike'),
    (42, 'spike'),
    (55, 'block'),
    (70, 'spike'),
    (85, 'double_spike'),
    (100, 'block'),
    (115, 'spike'),
    (130, 'spike'),
    (150, 'stair'),
    (170, 'block'),
    (190, 'spike'),
    (210, 'spike'),
    (230, 'block'),
    (260, 'spike'),
    (290, 'spike'),
    (320, 'block'),
    (350, 'spike'),
    (380, 'double_spike'),
    (420, 'block'),
    (460, 'spike'),
    (500, 'spike')
]
LEVEL_END = 550

# Flight zones (start, end)
flight_zones = [(180, 240), (360, 410)]

def ground_exists(dist):
    for start, end in flight_zones:
        if start < dist < end:
            return False
    return True

class Player:
    def __init__(self):
        self.y = GROUND_Y - PLAYER_SIZE
        self.vy = 0
        self.is_grounded = True
        self.distance = 0.0

    def jump(self):
        if self.is_grounded:
            self.vy = JUMP_SPEED
            self.is_grounded = False

    def update(self, hold=False):
        # Flight mode
        if not ground_exists(self.distance):
            if hold:
                self.vy += FLIGHT_THRUST
            self.vy += GRAVITY
            self.y += self.vy
            return

        # Ground mode
        if self.is_grounded and hold:
            self.vy = JUMP_SPEED
            self.is_grounded = False

        self.vy += GRAVITY
        self.y += self.vy

        if self.y >= GROUND_Y - PLAYER_SIZE:
            self.y = GROUND_Y - PLAYER_SIZE
            self.vy = 0
            self.is_grounded = True

    def draw(self, screen):
        """Draw the player as a yellow square."""
        pygame.draw.rect(screen, (255, 255, 0), (PLAYER_X, self.y, PLAYER_SIZE, PLAYER_SIZE))

    def get_rect(self):
        return pygame.Rect(PLAYER_X, self.y, PLAYER_SIZE, PLAYER_SIZE)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash Clone – Playable")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    def reset_game():
        nonlocal player, distance, running
        player = Player()
        distance = 0.0
        running = True

    player = Player()
    distance = 0.0
    running = True
    key_hold = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    player.jump()
                    key_hold = True
                if event.key == pygame.K_r:
                    reset_game()
            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    key_hold = False

        # Update
        player.distance = distance
        player.update(hold=key_hold)
        distance += SCROLL_SPEED

        # ---- Collision detection (rectangle-based) ----
        hit = False
        player_rect = player.get_rect()
        for d, typ in obstacles:
            rel = d - distance
            if 0 < rel < 30:
                x = PLAYER_X + rel * SCALE
                if typ == 'spike':
                    obs_rect = pygame.Rect(x - 8, GROUND_Y - 20, 16, 20)
                elif typ == 'block':
                    obs_rect = pygame.Rect(x - 10, GROUND_Y - 18, 20, 18)
                elif typ == 'double_spike':
                    obs_rect = pygame.Rect(x - 16, GROUND_Y - 20, 32, 20)
                elif typ == 'stair':
                    obs_rect = pygame.Rect(x - 10, GROUND_Y - 36, 34, 36)
                else:
                    continue
                if player_rect.colliderect(obs_rect):
                    hit = True
                    break

        if hit:
            print("💀 You died! Press R to restart.")
            while True:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                        reset_game()
                        break
                if running:
                    break
                clock.tick(10)

        if distance >= LEVEL_END:
            print("🎉 Level complete!")
            running = False

        # ---- Drawing ----
        screen.fill((30, 30, 40))

        # Ground
        if ground_exists(distance):
            pygame.draw.rect(screen, (100, 100, 100), (0, GROUND_Y, WIDTH, 5))
        else:
            for x in range(0, WIDTH, 20):
                pygame.draw.rect(screen, (80, 80, 90), (x, GROUND_Y, 10, 2))

        # Obstacles
        for d, typ in obstacles:
            rel = d - distance
            if 0 < rel < 300:
                x = PLAYER_X + rel * SCALE
                if x < WIDTH:
                    color = (255, 50, 50) if typ == 'spike' else (139, 69, 19)
                    if typ == 'spike':
                        half = 8
                        pygame.draw.polygon(screen, color, [
                            (x - half, GROUND_Y),
                            (x + half, GROUND_Y),
                            (x, GROUND_Y - 20)
                        ])
                    elif typ == 'block':
                        pygame.draw.rect(screen, color, (x - 10, GROUND_Y - 18, 20, 18))
                    elif typ == 'double_spike':
                        half = 8
                        pygame.draw.polygon(screen, color, [
                            (x - half - 8, GROUND_Y),
                            (x - 8, GROUND_Y),
                            (x - 8, GROUND_Y - 20)
                        ])
                        pygame.draw.polygon(screen, color, [
                            (x + 8, GROUND_Y),
                            (x + half + 8, GROUND_Y),
                            (x + 8, GROUND_Y - 20)
                        ])
                    elif typ == 'stair':
                        for i in range(3):
                            pygame.draw.rect(screen, color,
                                             (x - 10 + i*12, GROUND_Y - (i+1)*12, 10, 10))

        # Player
        player.draw(screen)

        # HUD
        text = font.render(f"Distance: {int(distance)}", True, (255,255,255))
        screen.blit(text, (10, 10))
        text2 = font.render("Space/Up to jump | Hold for flight | R to restart", True, (200,200,200))
        screen.blit(text2, (10, 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()