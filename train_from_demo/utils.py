import dxcam
import cv2
import numpy as np

class DxCamCapture:
    def __init__(self, region=None, target_fps=60):
        self.camera = dxcam.create()
        self.target_fps = target_fps
        self.region = region
        if region:
            self.camera.start(region=region, target_fps=target_fps)
        else:
            self.camera.start(target_fps=target_fps)

    def capture_frame(self):
        frame = self.camera.get_latest_frame()
        if frame is None:
            frame = self.camera.get_latest_frame()
        return frame

    def stop(self):
        self.camera.stop()
        del self.camera

def preprocess_frame(frame, target_size=(120,120)):
    if frame is None:
        return np.zeros(target_size, dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return resized / 255.0