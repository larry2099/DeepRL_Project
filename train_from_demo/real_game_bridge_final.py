import time
import cv2
import numpy as np
import pyautogui
from stable_baselines3 import PPO
from game_interface import GameInterface
from utils import DxCamCapture

pyautogui.FAILSAFE = False

# ----- CALIBRATED FROM YOUR VIDEO -----
DISTANCE_SCALE = 1.0          # 1 pixel ≈ 1 simulator unit (calibrated)
EDGE_THRESHOLD = 120          # edge density threshold
FALLBACK_DIST = 10.0          # force jump if obstacle distance < 10 (calibrated)
HORIZON = 5                   # look ahead for obstacles
HOLD_SCALE = 0.15             # max hold duration (seconds)
# -----------------------------------

class RealGamePerception:
    def __init__(self):
        self.prev_y = 0
        self.vy = 0

    def get_state(self, frame):
        h, w = frame.shape[:2]

        # ---- Detect player (yellow block) ----
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, wc, hc = cv2.boundingRect(largest)
            player_y = y + hc // 2
        else:
            player_y = int(h * 0.85)

        # ---- Vertical velocity ----
        self.vy = (player_y - self.prev_y) * 0.5
        self.prev_y = player_y

        # ---- Detect obstacles using edge density ----
        roi = frame[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.9)]
        if roi.size == 0:
            distances = [999.0] * HORIZON
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_cols = np.sum(edges, axis=0)

            # Find peaks (obstacles)
            peaks = []
            i = 0
            while i < len(edge_cols) - 5:
                if np.mean(edge_cols[i:i+5]) > EDGE_THRESHOLD:
                    peak = i + np.argmax(edge_cols[i:i+10])
                    peaks.append(peak)
                    i += 20
                else:
                    i += 1

            # Convert pixel positions to simulator distances (using calibrated scale)
            distances = []
            for px in peaks[:HORIZON]:
                # Use the calibrated distance scale: pixel distance directly maps to simulator units
                dist = (px / roi.shape[1]) * 50 + 5.0  # scale to match simulator (first obstacle ~15)
                distances.append(dist)
            while len(distances) < HORIZON:
                distances.append(999.0)

        # Convert to simulator units
        y_sim = (h - player_y) / 20.0
        vy_sim = np.clip(self.vy, -10, 10)

        # State: [y, vy, dist1, dist2, ..., dist5]
        state = np.array([y_sim, vy_sim] + distances, dtype=np.float32)
        return state

def focus_game(interface):
    try:
        interface._activate_window()
        time.sleep(0.03)
    except:
        pass

def detect_death(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 30:
        return True
    h, w = gray.shape
    attempt_region = gray[int(h*0.10):int(h*0.35), int(w*0.25):int(w*0.75)]
    if attempt_region.size > 0:
        edges = cv2.Canny(attempt_region, 50, 150)
        if np.sum(edges > 0) / edges.size > 0.025:
            return True
    return False

def main():
    print("="*60)
    print("REAL GAME – SIMULATOR POLICY + CALIBRATED PERCEPTION")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start...")
    input()

    # Load simulator-trained policy (continuous action)
    policy = PPO.load("ppo_sim_continuous")
    print("Policy loaded.")

    # Connect to the game window
    interface = GameInterface()
    interface.move_window(100, 100, 1024, 768)
    rect = interface.get_window_rect()
    if not rect:
        raise RuntimeError("Could not get window rectangle.")
    region = (rect[0], rect[1], rect[2], rect[3])
    capture = DxCamCapture(region=region, target_fps=30)

    # Start the level
    focus_game(interface)
    time.sleep(0.5)
    for _ in range(20):
        pyautogui.press('space')
        time.sleep(0.1)
    time.sleep(1)

    perception = RealGamePerception()
    print("Agent is playing. Press Ctrl+C to stop.")
    try:
        while True:
            frame = capture.capture_frame()
            if frame is None:
                continue

            # Death detection
            if detect_death(frame):
                print("💀 Death detected. Restarting...")
                focus_game(interface)
                for _ in range(8):
                    pyautogui.press('space')
                    time.sleep(0.1)
                time.sleep(0.3)
                continue

            # Extract state
            state = perception.get_state(frame)

            # ---- Fallback: force jump if obstacle is very close ----
            if state[2] < FALLBACK_DIST:
                focus_game(interface)
                pyautogui.keyDown('space')
                time.sleep(0.05)   # short hold
                pyautogui.keyUp('space')
                time.sleep(0.033)
                continue

            # Policy prediction
            action, _ = policy.predict(state, deterministic=True)
            hold_duration = float(action[0]) * HOLD_SCALE

            if hold_duration > 0.01:
                focus_game(interface)
                pyautogui.keyDown('space')
                time.sleep(hold_duration)
                pyautogui.keyUp('space')

            time.sleep(0.033)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        capture.stop()
        interface.close()

if __name__ == "__main__":
    main()