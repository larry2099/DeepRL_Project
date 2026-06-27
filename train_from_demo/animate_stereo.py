import pygame
import numpy as np
from stable_baselines3 import PPO
from simulator_stereo_madness import GeometryDashSimulator

# Pygame constants
WIDTH, HEIGHT = 800, 400
GROUND_Y = 350

# ----- FIXED SCALING -----
SCALE = 0.8          # reduced from 2.0 to keep player in view
PLAYER_X = 120       # fixed player position on screen
SPIKE_WIDTH = 12     # width of each spike triangle
# ------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash – Stereo Madness (Simulator)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    env = GeometryDashSimulator()
    model = PPO.load("ppo_stereo")

    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    running = True
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

        # ---- Draw ----
        screen.fill((30, 30, 40))

        # Ground
        pygame.draw.rect(screen, (100, 100, 100), (0, GROUND_Y, WIDTH, 5))

        # Draw obstacles (spikes, blocks, double spikes, stairs)
        for dist, typ, h in env.obstacles:
            rel_dist = dist - env.distance
            if 0 < rel_dist < 250:
                x = PLAYER_X + rel_dist * SCALE
                if x < WIDTH:
                    # Colors for each type
                    if typ == 'spike':
                        color = (255, 50, 50)
                    elif typ == 'block':
                        color = (139, 69, 19)
                    elif typ == 'double_spike':
                        color = (255, 100, 100)
                    elif typ == 'stair':
                        color = (150, 100, 50)
                    else:
                        color = (200, 200, 200)

                    if typ == 'spike':
                        half = SPIKE_WIDTH // 2
                        pygame.draw.polygon(screen, color, [
                            (x - half, GROUND_Y),
                            (x + half, GROUND_Y),
                            (x, GROUND_Y - 20)
                        ])
                    elif typ == 'block':
                        pygame.draw.rect(screen, color, (x - 10, GROUND_Y - 18, 20, 18))
                    elif typ == 'double_spike':
                        half = SPIKE_WIDTH // 2
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

        # ---- Draw Player (yellow square) ----
        # Ensure player stays on ground when y=0, and clamp to screen bottom
        player_y = GROUND_Y - env.y * SCALE
        if player_y > GROUND_Y:
            player_y = GROUND_Y
        # Keep player within screen bounds (just in case)
        if player_y < 0:
            player_y = 0
        pygame.draw.rect(screen, (255, 255, 0), (PLAYER_X - 8, player_y - 8, 16, 16))

        # HUD
        text1 = font.render(f"Distance: {env.distance:.1f}", True, (255,255,255))
        text2 = font.render(f"Steps: {steps}   Reward: {total_reward:.1f}", True, (255,255,255))
        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 30))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    # Show completion message
    if done and env.distance >= env.level_end:
        pygame.time.wait(500)
        screen.fill((30, 30, 40))
        text = font.render("🎉 LEVEL COMPLETE!", True, (0, 255, 0))
        screen.blit(text, (WIDTH//2 - 100, HEIGHT//2))
        pygame.display.flip()
        pygame.time.wait(3000)

    pygame.quit()

if __name__ == "__main__":
    main()