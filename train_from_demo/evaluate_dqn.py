from game_env import GeometryDashEnv
from dqn_agent_video import DQNAgent
import numpy as np
import os

def evaluate_dqn():
    print("="*60)
    print("EVALUATE DQN AGENT")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start evaluation...")
    input()

    # ---- MATCH TRAINING PARAMETERS ----
    env = GeometryDashEnv(
        target_size=(84,84),
        frame_skip=3,          # same as training
        max_steps=2500
    )

    agent = DQNAgent(state_shape=(4,84,84), num_actions=2)

    # Load best model
    model_paths = ["dqn_finetuned_best.pth", "dqn_best.pth", "dqn_final.pth"]
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            agent.load(path)
            print(f"✅ Loaded model from {path}")
            loaded = True
            break
    if not loaded:
        print("❌ No trained model found. Please train first with train_dqn_video.py")
        return

    # Allow a tiny bit of exploration to avoid deterministic failures
    agent.epsilon = 0.05

    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0
    steps = 0

    print("\n🤖 Agent is playing (epsilon=0.05). Press Ctrl+C to stop.\n")
    try:
        while not done and steps < env.max_steps:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, _, _ = env.step(action)
            state = np.array(state, dtype=np.float32)
            total_reward += reward
            steps += 1
            if steps % 100 == 0:
                print(f"Steps: {steps}, Reward: {total_reward:.2f}")
        print(f"\n🏁 Episode finished. Total reward: {total_reward:.2f} | Steps: {steps}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate_dqn()