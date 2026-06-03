"""
Record a demonstration of you playing Geometry Dash.
Press SPACE to start the level – recording begins automatically.
Press ESC to stop recording (or run for max 60 seconds).
Saves frames and actions to 'demonstration.npz'.
"""

import time
import numpy as np
import cv2
import pyautogui
import keyboard
from game_interface import GameInterface
from utils import DxCamCapture, preprocess_frame

def record_demo(max_duration=60, target_size=(84,84), frame_skip=4):
    print("="*60)
    print("RECORDING DEMONSTRATION")
    print("="*60)
    print("1. Open Geometry Dash manually (windowed mode).")
    print("2. Select the first level (Stereo Madness) but do NOT start it.")
    print("3. Press SPACE to start the level – recording will begin automatically.")
    print("4. Play normally (use SPACE to jump).")
    print("5. Recording stops after {} seconds or when you press ESC.".format(max_duration))
    print("-"*60)

    # Attach to the already‑open game window
    game = GameInterface()
    rect = game.get_window_rect()
    if not rect:
        raise RuntimeError("Could not find game window.")
    region = (rect[0], rect[1], rect[2], rect[3])

    capture = DxCamCapture(region=region, target_fps=60)

    # Wait for user to press space to start recording
    print("Waiting for SPACE to start recording...")
    keyboard.wait('space')
    print("Recording started! Play now.\n")

    frames = []      # preprocessed frames (single, not stacked)
    actions = []     # action at each step (0 or 1)
    step = 0
    start_time = time.time()
    interval = 0.016 * frame_skip   # time per step (~0.064 sec)

    while (time.time() - start_time) < max_duration:
        # Capture current frame
        frame = capture.capture_frame()
        if frame is None:
            continue
        processed = preprocess_frame(frame, target_size)   # shape (84,84)

        # Check if space is currently pressed (global)
        space_pressed = keyboard.is_pressed('space')
        action = 1 if space_pressed else 0

        frames.append(processed)
        actions.append(action)
        step += 1

        if step % 100 == 0:
            print(f"Recorded {step} steps...")

        time.sleep(interval)

        # Stop if ESC pressed
        if keyboard.is_pressed('esc'):
            print("ESC pressed. Stopping recording.")
            break

    capture.stop()
    game.close()

    # Save as npz
    np.savez("demonstration.npz",
             frames=np.array(frames, dtype=np.float32),
             actions=np.array(actions, dtype=np.int8))
    print(f"\nRecording finished. Saved {step} steps to demonstration.npz")
    print("File size:", frames[0].shape, "x", len(frames))

if __name__ == "__main__":
    record_demo()