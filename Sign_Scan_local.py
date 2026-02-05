import os
import cv2
from gtts import gTTS
import pygame
import time
from ultralytics import YOLO

class ObjectDetectionWithSound:
    def __init__(self, model_path, detection_interval=5, conf_threshold=0.6):
        self.model = YOLO(model_path)
        self.detection_interval = detection_interval
        self.conf_threshold = conf_threshold
        self.last_detection_time = {}
        pygame.mixer.init()

    def play_sound_for_class(self, class_name):
        """Converts class name to speech and plays the audio."""
        audio_file = f"{class_name.replace(' ', '_')}.mp3"

        # Generate audio only once per class
        if not os.path.exists(audio_file):
            tts = gTTS(text=f"Alert: {class_name} detected!", lang='en', slow=False)
            tts.save(audio_file)

        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

    def detect_and_play_sound(self, source=0):
        """Detect objects and display results with sound alerts."""
        print("Starting object detection...")
        camera = cv2.VideoCapture(source)

        if not camera.isOpened():
            print("Error: Could not open video source.")
            return

        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Run YOLOv8 inference
                results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)

                # Annotate frame with results
                annotated_frame = results[0].plot()  # draws bounding boxes and labels

                # Show the frame in a window
                cv2.imshow("Object Detection with Sound", annotated_frame)

                # Process detections
                detections = results[0].boxes
                for box in detections:
                    class_index = int(box.cls[0])
                    class_name = self.model.names[class_index]
                    current_time = time.time()

                    if class_name not in self.last_detection_time or \
                       (current_time - self.last_detection_time[class_name]) > self.detection_interval:
                        self.last_detection_time[class_name] = current_time
                        print(f"Detected: {class_name}")
                        self.play_sound_for_class(class_name)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break

        except Exception as e:
            print(f"Error during detection: {e}")
        finally:
            camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetectionWithSound(
        model_path="best.pt",   # Replace with your actual model
        detection_interval=5,   # 5 seconds per repeated class alert
        conf_threshold=0.6
    )
    detector.detect_and_play_sound(source=0)
