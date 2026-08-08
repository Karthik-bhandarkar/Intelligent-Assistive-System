import os
import sys
import argparse
import cv2
import time
import requests
import numpy as np

# Ensure the repo root is importable regardless of launch location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection import yolo_sign_detection
from src.tts import tts_audio

class ObjectDetectionWithSound:
    def __init__(self, detection_interval=5, conf_threshold=0.6):
        self.detection_interval = detection_interval
        self.conf_threshold = conf_threshold
        self.last_detection_time = {}

    def fetch_frame(self, source):
        """
        Fetch a single frame from ESP32 camera URL or local webcam.
        """
        if isinstance(source, str) and source.startswith("http"):
            try:
                response = requests.get(source, stream=True, timeout=5)
                if response.status_code == 200:
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return frame
                else:
                    print(f"Failed to fetch frame. HTTP {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error fetching frame: {e}")
                return None
        else:
            # Handle webcam input
            if not hasattr(self, "camera") or self.camera is None:
                cam_id = 0
                if str(source).isdigit():
                    cam_id = int(source)
                self.camera = cv2.VideoCapture(cam_id)
            ret, frame = self.camera.read()
            return frame if ret else None

    def detect_and_play_sound(self, source):
        """Main detection loop for both webcam and ESP32 streams."""
        print(f"Starting object detection (source: {source})...")

        try:
            while True:
                frame = self.fetch_frame(source)
                if frame is None:
                    print("No frame received. Retrying...")
                    time.sleep(0.2)
                    continue

                # Run YOLO inference via src module
                best_class, annotated_frame = yolo_sign_detection.detect_sign(
                    frame,
                    conf_threshold=self.conf_threshold,
                    model_type='sign'
                )

                if annotated_frame is not None:
                    cv2.imshow("Object Detection with Sound", annotated_frame)
                else:
                    cv2.imshow("Object Detection with Sound", frame)

                # Process detected class
                if best_class:
                    current_time = time.time()
                    if (best_class not in self.last_detection_time or
                            (current_time - self.last_detection_time[best_class]) > self.detection_interval):
                        self.last_detection_time[best_class] = current_time
                        print(f"Detected: {best_class}")
                        tts_audio.speak(f"Alert: {best_class} detected!", 'en')

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting detection...")
                    break

        except Exception as e:
            print(f"Error during detection: {e}")

        finally:
            if hasattr(self, "camera") and self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Board Detection CLI")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: 'webcam', '0', or ESP32 URL (e.g. http://10.219.6.122/cam-hi.jpg)"
    )
    args = parser.parse_args()

    detector = ObjectDetectionWithSound(detection_interval=5, conf_threshold=0.6)

    cam_source = args.source
    if cam_source is None:
        # Default to ESP32 camera URL if unspecified
        cam_source = "http://10.219.6.122/cam-hi.jpg"
    elif cam_source.lower() == "webcam":
        cam_source = 0

    detector.detect_and_play_sound(source=cam_source)
