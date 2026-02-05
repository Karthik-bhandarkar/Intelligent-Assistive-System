import os
import cv2
import time
import pygame
import requests
import numpy as np
from gtts import gTTS
from ultralytics import YOLO


class ObjectDetectionWithSound:
    def __init__(self, model_path, detection_interval=5, conf_threshold=0.75):
        self.model = YOLO(model_path)
        self.detection_interval = detection_interval
        self.conf_threshold = conf_threshold
        self.last_detection_time = {}
        pygame.mixer.init()

    def play_sound_for_class(self, class_name):
        """Convert class name to speech and play it."""
        audio_file = f"{class_name.replace(' ', '_')}.mp3"

        # Generate and cache audio once
        if not os.path.exists(audio_file):
            tts = gTTS(text=f"Alert: {class_name} detected!", lang='en', slow=False)
            tts.save(audio_file)

        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

    def fetch_frame(self, source):
        """
        Fetch a single frame from ESP32 camera safely or from local webcam.
        """
        if isinstance(source, str) and source.startswith("http"):
            try:
                response = requests.get(source, stream=True, timeout=5)
                if response.status_code == 200:
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("Failed to decode frame (corrupt data).")
                        return None
                        
                    return frame
                else:
                    print(f"Failed to fetch frame. HTTP {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error fetching frame: {e}")
                return None
        else:
            # Handle webcam input
            if not hasattr(self, "camera"):
                self.camera = cv2.VideoCapture(int(source))
            ret, frame = self.camera.read()
            return frame if ret else None

    def detect_and_play_sound(self, source):
        """Main detection loop for both webcam and ESP32 streams."""
        print("Starting object detection...")

        try:
            while True:
                frame = self.fetch_frame(source)
                if frame is None:
                    print("No frame received. Retrying...")
                    time.sleep(0.2)
                    continue

                # Run YOLO inference
                results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
                detections = results[0].boxes
                annotated_frame = results[0].plot()

                # Display annotated frame
                cv2.imshow("Object Detection with Sound", annotated_frame)

                # Process detected objects
                for box in detections:
                    class_index = int(box.cls[0])
                    class_name = self.model.names[class_index]
                    current_time = time.time()

                    if (class_name not in self.last_detection_time or
                            (current_time - self.last_detection_time[class_name]) > self.detection_interval):
                        self.last_detection_time[class_name] = current_time
                        print(f"Detected: {class_name}")
                        self.play_sound_for_class(class_name)

                # Exit loop when 'q' pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting detection...")
                    break

        except Exception as e:
            print(f"Error during detection: {e}")

        finally:
            if hasattr(self, "camera"):
                self.camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetectionWithSound(
        model_path="best.pt",
        detection_interval=5,
        conf_threshold=0.6
    )

    # To use local webcam → source = 0
    # To use ESP32 camera → source = "http://<your_esp32_ip>/cam-hi.jpg"
    cam_source = "http://10.219.6.122/cam-hi.jpg"

    detector.detect_and_play_sound(source=cam_source)
