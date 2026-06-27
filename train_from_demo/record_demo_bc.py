import time
import cv2
import numpy as np
import pyautogui
import keyboard
from game_interface import GameInterface
from utils import DxCamCapture, preprocess_frame

pyautogui.FAILSAFE = False

def record_demo(target_size=(120,120), frame_skip=4, max_duration=180):
    print("="*60)
    print("RECORD DEMONSTRATION FOR BC")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press SPACE to start the level – recording begins automatically.")
    print("3. Play normally (jump using SPACE or UP arrow).")
    print("4. Press ESC to stop recording (or wait {} seconds).".format(max_duration))
    print("-"*60)

    interface = GameInterface()
    interface.move_window(100, 100, 1024, 768)
    rect = interface.get_window_rect()
    if not rect:
        raise RuntimeError("Could not get window rectangle.")
    region = (rect[0], rect[1], rect[2], rect[3])
    capture = DxCamCapture(region=region, target_fps=30)

    print("Waiting for SPACE to start recording...")
    keyboard.wait('space')
    print("Recording started! Play now.\n")

    frames = []
    actions = []
    step = 0
    start_time = time.time()
    interval = 0.016 * frame_skip
    press_counter = 0

    while (time.time() - start_time) < max_duration:
        frame = capture.capture_frame()
        if frame is None:
            continue
        processed = preprocess_frame(frame, target_size)
        # Detect jump keys (space or up arrow)
        space_pressed = keyboard.is_pressed('space') or keyboard.is_pressed('up')
        if space_pressed:
            press_counter += 1
        else:
            press_counter = 0
        # Action: 0 = none, 1 = short tap, 2 = hold (pressed for ≥3 frames)
        if press_counter >= 3:
            action = 2
        elif space_pressed:
            action = 1
        else:
            action = 0
        frames.append(processed)
        actions.append(action)
        step += 1
        if step % 100 == 0:
            print(f"Recorded {step} steps...")
        time.sleep(interval)
        if keyboard.is_pressed('esc'):
            print("ESC pressed. Stopping recording.")
            break

    capture.stop()
    interface.close()

    np.savez("demo_bc.npz",
             frames=np.array(frames, dtype=np.float32),
             actions=np.array(actions, dtype=np.int8))
    print(f"\nRecording finished. Saved {step} steps to demo_bc.npz")

if __name__ == "__main__":
    record_demo()