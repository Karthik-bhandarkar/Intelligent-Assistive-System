import requests
import numpy as np
import cv2
from PIL import Image

def get_frame(url):
    """
    Fetch a single frame from ESP32 camera, decode safely, and return as PIL Image (RGB).
    """
    try:
        response = requests.get(url, stream=True, timeout=2)
        if response.status_code == 200:
            # 1. Safely read bytes
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            
            # 2. Decode using imdecode (handles corrupt frames gracefully most of the time)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                # print("Failed to decode frame (corrupt data).")
                return None

            # Rotate 180 degrees (Flip upside down) as requested ("rotate 90 again")
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Apply standard resolution (640x480)
            frame = cv2.resize(frame, (640, 480))

            # 3. Convert BGR (OpenCV) -> RGB (PIL/BLIP expectation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 4. Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            return pil_image
        else:
            print(f"Failed to fetch frame. HTTP {response.status_code}")
            return None
    except Exception as e:
        # print(f"Error fetching frame: {e}")
        return None
