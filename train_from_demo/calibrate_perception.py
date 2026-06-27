import cv2
import numpy as np
import os

def calibrate_distance(video_path="geometry_dash_run.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Video not found.")
        return

    # We'll manually find the player's position and obstacle positions
    # For simplicity, we'll use the first obstacle distance in pixels.
    # Run the video and when the first obstacle is about to hit, note the pixel distance from the player.
    # We'll automate: detect player and obstacles using simple methods.

    player_positions = []
    obstacle_positions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # Detect player (yellow block)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, wc, hc = cv2.boundingRect(largest)
            player_x = x + wc // 2
            player_y = y + hc // 2
        else:
            player_x = 0
            player_y = 0

        # Detect obstacle (spike) using edge density in the region ahead
        roi = frame[int(h*0.7):int(h*0.9), int(w*0.4):int(w*0.9)]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_cols = np.sum(edges, axis=0)
            for col in range(edge_cols.shape[0]):
                if np.mean(edge_cols[col:col+5]) > 150:
                    obstacle_x = col + int(w*0.4)
                    break
            else:
                obstacle_x = None
        else:
            obstacle_x = None

        if obstacle_x is not None and player_x > 0:
            pixel_dist = obstacle_x - player_x
            if 0 < pixel_dist < 200:  # only close obstacles
                # We know the simulator distance at which the first obstacle appears (about 15 units)
                # We'll store the mapping: pixel_dist -> simulator_dist
                # We'll use the known simulator distance for the first spike (15)
                # But we need to calibrate the scale.
                # We'll compute the scale by assuming that when pixel_dist is minimal (the obstacle is just about to hit),
                # the simulator distance is ~5 (the fallback distance).
                # We'll store multiple points and fit a line.
                # For simplicity, we'll just output the pixel_dist when the obstacle is close.
                print(f"Frame {frame_count}: pixel distance to obstacle: {pixel_dist}")
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    calibrate_distance()