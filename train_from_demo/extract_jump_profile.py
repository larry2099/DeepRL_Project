import numpy as np
import cv2

def extract_jump_profile():
    data = np.load("demo_bc.npz")  # from recorded demo
    frames = data["frames"]        # (N, 120, 120) grayscale
    actions = data["actions"]      # (N,) 0,1,2

    # Estimate distance traveled from frame differences
    # We'll use optical flow or simple cross-correlation to estimate horizontal shift
    # Simpler: use the player's x position (detected by yellow blob) to compute distance
    # But we don't have player x in the demo (only frames). We'll compute distance by tracking the screen movement.
    # We'll use a simple method: detect edges and track horizontal shift.

    prev_gray = None
    distance = 0.0
    jump_positions = []  # list of (distance, action)

    for i, (frame, action) in enumerate(zip(frames, actions)):
        gray = (frame * 255).astype(np.uint8)
        if prev_gray is not None:
            # Compute optical flow or cross-correlation to estimate horizontal shift
            # We'll use phase correlation to find the shift between frames
            # But this is complex. Instead, we'll just use the frame index as a proxy for distance
            # Since the game scrolls at constant speed, frame index is proportional to distance.
            # That's not accurate but good enough for a proof-of-concept.
            pass
        # We'll store action at each frame index
        # Later, in real game, we'll use frame index as distance.
        # But we need to know the frame rate and scroll speed.
        # We'll estimate scroll speed from the video.
        # Since we have the MP4, we can also extract distance from the video directly.

        # Actually, we have the MP4. We can extract distance from the video using player's x position
        # But we already have a script that extracts frames and actions from the video.
        # Let's just use the video_demo.npz which contains frames and actions.
        # We'll compute distance for each frame in the video by detecting the player's x position.
        # We'll then create a mapping from distance to action.

    # Simpler: We'll read the video directly and track the player's x position.
    cap = cv2.VideoCapture("geometry_dash_run.mp4")
    if not cap.isOpened():
        print("Video not found. Using demo_bc.npz indices as distance.")
        # Fallback: use frame index as distance
        distances = np.arange(len(actions))
        jump_profile = [(d, a) for d, a in zip(distances, actions) if a != 0]
        np.savez("jump_profile.npz", distances=distances, actions=actions)
        return

    # Track player's x position in video
    player_x_positions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Detect player (yellow blob)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            player_x = x + w // 2
        else:
            player_x = 0
        player_x_positions.append(player_x)
    cap.release()

    # Convert x positions to distance (cumulative movement)
    distances = []
    cum_dist = 0.0
    for i in range(1, len(player_x_positions)):
        # Movement is negative when the player moves left? Actually the player stays roughly fixed, obstacles move.
        # Better: use frame index as distance (since scroll speed is constant)
        # But we have the actions from video_demo.npz (extracted earlier).
        # Let's load video_demo.npz actions and use frame index as distance.
        pass

    # The simplest: use frame index as distance
    # We'll load video_demo.npz actions
    try:
        data = np.load("video_demo.npz")
        actions = data["actions"]
        distances = np.arange(len(actions))
        jump_profile = [(d, a) for d, a in zip(distances, actions) if a != 0]
        np.savez("jump_profile.npz", distances=distances, actions=actions)
        print(f"Created jump profile with {len(jump_profile)} jump events")
    except:
        print("video_demo.npz not found. Using demo_bc.npz.")
        data = np.load("demo_bc.npz")
        actions = data["actions"]
        distances = np.arange(len(actions))
        np.savez("jump_profile.npz", distances=distances, actions=actions)

if __name__ == "__main__":
    extract_jump_profile()