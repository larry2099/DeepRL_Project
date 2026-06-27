import pygame
import numpy as np
from stable_baselines3 import PPO
from simulator_corrected import GeometryDashSimulator

# Pygame constants
WIDTH, HEIGHT = 800, 400
GROUND_Y = 350
SCALE = 2.0          # pixels per simulator unit (larger = more spaced)
PLAYER_X = 120       # fixed player position on screen
SPIKE_WIDTH = 12     # width of each spike triangle

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash Simulator – Agent Playing")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    env = GeometryDashSimulator()
    model = PPO.load("ppo_sim_corrected")

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

        # Draw obstacles (spikes pointing upward)
        for obs_dist in env.obstacles:
            rel_dist = obs_dist - env.distance
            if 0 < rel_dist < 250:   # draw obstacles within 250 units ahead
                x = PLAYER_X + rel_dist * SCALE
                if x < WIDTH:
                    # Spike triangle: base at ground, apex up
                    half = SPIKE_WIDTH // 2
                    pygame.draw.polygon(screen, (255, 50, 50), [
                        (x - half, GROUND_Y),
                        (x + half, GROUND_Y),
                        (x, GROUND_Y - 25)
                    ])

        # Draw player (yellow square)
        player_y = GROUND_Y - env.y * SCALE
        if player_y > GROUND_Y:
            player_y = GROUND_Y
        pygame.draw.rect(screen, (255, 255, 0), (PLAYER_X - 8, player_y - 8, 16, 16))

        # HUD
        text1 = font.render(f"Distance: {env.distance:.1f}", True, (255,255,255))
        text2 = font.render(f"Steps: {steps}   Reward: {total_reward:.1f}", True, (255,255,255))
        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 30))

        pygame.display.flip()
        clock.tick(30)

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