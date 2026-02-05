from ultralytics import YOLO
from gtts import gTTS
import pygame
import time
import os
import cv2

# Initialize pygame for sound playback
pygame.mixer.init()

# Load YOLO model
model = YOLO("best.onnx")

# Function to play a sound for a detected class
def play_sound_for_class(class_name):
    """
    Converts class name to speech and plays the audio.
    """
    audio_file = f"{class_name.replace(' ', '_')}.mp3"

    # Generate the audio if it doesn't already exist
    if not os.path.exists(audio_file):
        tts = gTTS(text=f"Alert: {class_name} detected!", lang='en', slow=False)
        tts.save(audio_file)

    # Play the sound
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Wait for the sound to finish
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Detection function
def detect_and_play_sound(model, source, conf=0.6):
    """
    Detect objects in a video source and play sound for detected classes.
    """
    last_detection_time = {}  # Track the last detection time for each class
    detection_interval = 5  # Minimum interval (in seconds) between detections of the same class

    # Initialize the video source
    if isinstance(source, str):  # IP camera URL
        camera = cv2.VideoCapture(source)
    else:  # Local webcam
        camera = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = camera.read()
            if ret:
                results = model.predict(source=frame, conf=conf, show=False)
                detections = results[0].boxes

                # Process detections
                for box in detections:
                    class_index = int(box.cls[0])  # Class index
                    class_name = model.names[class_index]  # Get class name
                    current_time = time.time()

                    # Check if it's time to alert for this class again
                    if class_name not in last_detection_time or (current_time - last_detection_time[class_name] > detection_interval):
                        last_detection_time[class_name] = current_time
                        print(f"Detected: {class_name}")
                        play_sound_for_class(class_name)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

# Run detection
detect_and_play_sound(model=model, source=0, conf=0.6)  # Use 0 for local webcam or provide an IP camera URL
