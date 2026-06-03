from game_env import GeometryDashEnv
import cv2
import time

env = GeometryDashEnv()
time.sleep(2)
frame = env.capture.capture_frame()
if frame is not None:
    cv2.imshow("Game capture", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to capture frame.")
env.close()