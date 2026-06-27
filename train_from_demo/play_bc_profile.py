import time
import cv2
import numpy as np
import pyautogui
import math
from collections import deque
from game_interface import GameInterface
from utils import DxCamCapture, preprocess_frame

pyautogui.FAILSAFE = False

def estimate_distance(frame, prev_gray, initial_shift=0):
    """
    Estimate the horizontal distance traveled using phase correlation.
    Returns cumulative distance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, gray

    # Use phase correlation to estimate shift
    # We'll crop the middle region to avoid edges
    h, w = gray.shape
    crop_h = h // 2
    crop_w = w // 2
    roi1 = prev_gray[h//4:3*h//4, w//4:3*w//4]
    roi2 = gray[h//4:3*h//4, w//4:3*w//4]
    # Compute shift using phase correlation
    # But this is heavy. We'll use a simpler method: track edge density shift.
    # For a quick proof, we'll just use frame index increment.
    # We'll assume constant scroll speed, so distance is proportional to number of frames.
    # This is a rough approximation but may work.

    # Actually, we can use optical flow to compute the horizontal movement.
    # But to keep it simple, we'll just increment distance by a fixed amount per frame.
    # The scroll speed is constant in Geometry Dash.
    # We can calibrate by measuring the speed from the video.
    # For now, we'll use a fixed increment per frame.
    return 0.0, gray

def main():
    print("="*60)
    print("PLAY USING JUMP PROFILE")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start...")
    input()

    # Load jump profile
    try:
        data = np.load("jump_profile.npz")
        distances = data["distances"]
        actions = data["actions"]
        # Build a mapping from distance to action (we'll use interpolation)
        # We'll use the distance and action arrays as reference.
        # In the real game, we'll compute distance and use the nearest action in the profile.
        print(f"Loaded jump profile: {len(distances)} frames")
    except:
        print("No jump profile found. Run extract_jump_profile.py first.")
        return

    interface = GameInterface()
    interface.move_window(100, 100, 1024, 768)
    rect = interface.get_window_rect()
    region = (rect[0], rect[1], rect[2], rect[3])
    capture = DxCamCapture(region=region, target_fps=30)

    # Start the level
    interface._activate_window()
    time.sleep(0.5)
    for _ in range(20):
        pyautogui.press('space')
        time.sleep(0.1)
    time.sleep(1)

    # We'll use frame count as distance (since scroll speed is constant)
    frame_count = 0
    prev_gray = None
    distance = 0.0

    print("Agent playing using jump profile. Press Ctrl+C to stop.\n")
    try:
        while True:
            frame = capture.capture_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                # Use cross-correlation to estimate horizontal shift
                # We'll use a simpler method: just increment distance by a fixed amount
                # We can calibrate the scroll speed from the video: distance per frame
                # For now, we'll use a constant increment
                scroll_speed = 0.8  # pixels per frame (approximate)
                # To get accurate distance, we would need to calibrate.
                # For simplicity, we'll use frame count.
                distance += 0.1  # arbitrary unit
            prev_gray = gray

            frame_count += 1
            # Find the nearest distance in the profile and get action
            idx = np.argmin(np.abs(distances - distance))
            action = actions[idx]

            # Map action to hold duration
            if action == 2:
                hold_duration = 0.06
            elif action == 1:
                hold_duration = 0.02
            else:
                hold_duration = 0.0

            if hold_duration > 0:
                interface._activate_window()
                pyautogui.keyDown('space')
                time.sleep(hold_duration)
                pyautogui.keyUp('space')

            time.sleep(0.033)
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}, distance: {distance:.1f}, action: {action}")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        capture.stop()
        interface.close()

if __name__ == "__main__":
    main()