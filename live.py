from ultralytics import YOLO
import cv2
import requests
import numpy as np

def fetch_frame(source):
    """
    Fetch a single frame from the given ESP32 camera URL.

    :param source: URL of the ESP32 camera stream (e.g., "http://192.168.0.113/cam-hi.jpg")
    :return: Decoded frame as a NumPy array, or None if fetching failed.
    """
    try:
        response = requests.get(source, stream=True)
        if response.status_code == 200:
            # Convert the response content to a NumPy array
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            # Decode the image
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        else:
            print(f"Failed to fetch frame. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while fetching frame: {e}")
        return None

def detect_and_display(model, source, conf=0.6):
    """
    Continuously fetch frames from the ESP32 camera, run YOLO detection, and display results.

    :param model: YOLO model object for making predictions.
    :param source: URL of the ESP32 camera stream (e.g., "http://192.168.0.113/cam-hi.jpg")
    :param conf: Confidence threshold for YOLO predictions.
    """
    try:
        while True:
            # Fetch the frame from the camera
            frame = fetch_frame(source)
            if frame is not None:
                # Run YOLO model prediction on the frame
                results = model.predict(source=frame, conf=conf, show=True)
                # Display the results (YOLO handles the annotated display with show=True)
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received. Retrying...")
    except Exception as e:
        print(f"Error occurred during detection: {e}")
    finally:
        cv2.destroyAllWindows()

# Initialize the YOLO model
model = YOLO("best.pt")

# Replace with your ESP32 camera URL
cam_url = "http://192.168.198.155/cam-hi.jpg"

# Start live detection
detect_and_display(model=model, source=cam_url, conf=0.6)
