import cv2
import numpy as np

def extract_actions_from_video(video_path="geometry_dash_run.mp4", output_npz="video_demo.npz", target_size=(120,120)):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    frames = []
    actions = []
    prev_y = None
    jump_threshold = 4.0
    hold_threshold = 8.0  # if player moves up more than this, it's a hold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect player (yellow block) using HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            player_y = y + h // 2
        else:
            player_y = None

        # Preprocess frame for CNN
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        frames.append(normalized)

        # Determine action: 0 = no jump, 1 = short jump, 2 = hold (continuous jump)
        if prev_y is not None and player_y is not None:
            dy = prev_y - player_y
            if dy > hold_threshold:
                action = 2
            elif dy > jump_threshold:
                action = 1
            else:
                action = 0
        else:
            action = 0
        actions.append(action)
        prev_y = player_y

    cap.release()
    print(f"Extracted {len(frames)} frames")
    print(f"Actions: 0={actions.count(0)}, 1={actions.count(1)}, 2={actions.count(2)}")
    np.savez(output_npz, frames=np.array(frames, dtype=np.float32), actions=np.array(actions, dtype=np.int8))
    print(f"Saved to {output_npz}")

if __name__ == "__main__":
    # Place your MP4 file in the same folder as this script
    extract_actions_from_video("geometry_dash_run.mp4")