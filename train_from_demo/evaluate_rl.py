from stable_baselines3 import PPO
from game_env import GeometryDashEnv

def main():
    print("="*60)
    print("EVALUATE – Trained Agent on Real Game")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start evaluation...")
    input()

    env = GeometryDashEnv()
    model = PPO.load("ppo_real_game")

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    print("\n🤖 Agent is playing. Press Ctrl+C to stop.\n")
    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            if steps % 100 == 0:
                print(f"Steps: {steps}, Current reward: {total_reward:.2f}")
        print(f"\n🏁 Episode finished. Total reward: {total_reward:.2f} | Steps: {steps}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()