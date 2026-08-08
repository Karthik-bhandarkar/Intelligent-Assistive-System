import requests
import numpy as np
import cv2
from PIL import Image

_cam = None

def get_frame(url):
    """
    Fetch a single frame from ESP32 camera URL or local webcam, decode safely, and return as PIL Image (RGB).
    """
    global _cam
    try:
        if isinstance(url, str) and url.startswith("http"):
            response = requests.get(url, stream=True, timeout=2)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    return None
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            else:
                return None
        else:
            # Handle webcam input ('webcam', 0, etc.)
            if _cam is None or not _cam.isOpened():
                cam_id = 0
                if isinstance(url, int):
                    cam_id = url
                elif isinstance(url, str) and url.isdigit():
                    cam_id = int(url)
                _cam = cv2.VideoCapture(cam_id)
            
            ret, frame = _cam.read()
            if ret and frame is not None:
                frame_resized = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            return None
    except Exception as e:
        return None
