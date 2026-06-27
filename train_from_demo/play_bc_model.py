import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import pyautogui
from collections import deque
from game_interface import GameInterface
from utils import DxCamCapture, preprocess_frame

pyautogui.FAILSAFE = False

class DeepCNN4Channels(nn.Module):
    def __init__(self, input_channels=4, num_actions=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, 120, 120)
            n_flatten = self.conv(sample).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features)

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

def focus_game(interface):
    try:
        interface._activate_window()
        time.sleep(0.03)
    except:
        pass

def main():
    print("="*60)
    print("PLAY REAL GAME – BC (STACKED FRAMES + HOLD)")
    print("="*60)
    print("1. Open Geometry Dash (windowed, Level 1 selected).")
    print("2. Press Enter to start...")
    input()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCNN4Channels(input_channels=4, num_actions=3).to(device)
    model.load_state_dict(torch.load("bc_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded.")

    interface = GameInterface()
    interface.move_window(100, 100, 1024, 768)
    rect = interface.get_window_rect()
    region = (rect[0], rect[1], rect[2], rect[3])
    capture = DxCamCapture(region=region, target_fps=30)

    focus_game(interface)
    time.sleep(0.5)
    for _ in range(20):
        pyautogui.press('space')
        time.sleep(0.1)
    time.sleep(1)

    frame_buffer = deque(maxlen=4)
    action_buffer = []  # for smoothing
    SMOOTHING = 3       # majority vote over last 3 predictions

    print("🤖 Agent is playing (BC). Press Ctrl+C to stop.\n")
    steps = 0
    try:
        while True:
            frame = capture.capture_frame()
            if frame is None:
                continue

            if detect_death(frame):
                print("💀 Death detected. Restarting...")
                focus_game(interface)
                for _ in range(8):
                    pyautogui.press('space')
                    time.sleep(0.1)
                time.sleep(0.3)
                frame_buffer.clear()
                action_buffer.clear()
                continue

            processed = preprocess_frame(frame, target_size=(120,120))
            frame_buffer.append(processed)
            if len(frame_buffer) < 4:
                time.sleep(0.033)
                continue

            stacked = np.stack(list(frame_buffer), axis=0)
            state = torch.FloatTensor(stacked).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(state)
                probs = torch.softmax(logits, dim=1)[0]
                action = torch.argmax(probs).item()

            # Smoothing: majority vote over last SMOOTHING predictions
            action_buffer.append(action)
            if len(action_buffer) > SMOOTHING:
                action_buffer.pop(0)
            if len(action_buffer) == SMOOTHING:
                # majority vote
                smoothed_action = max(set(action_buffer), key=action_buffer.count)
            else:
                smoothed_action = action

            # Map action to hold duration (adjusted for better timing)
            if smoothed_action == 2:
                hold_duration = 0.07  # longer hold for flying
            elif smoothed_action == 1:
                hold_duration = 0.025 # short tap
            else:
                hold_duration = 0.0

            if hold_duration > 0:
                focus_game(interface)
                pyautogui.keyDown('space')
                time.sleep(hold_duration)
                pyautogui.keyUp('space')

            steps += 1
            if steps % 100 == 0:
                print(f"Steps: {steps}")
            time.sleep(0.033)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        capture.stop()
        interface.close()

if __name__ == "__main__":
    main()