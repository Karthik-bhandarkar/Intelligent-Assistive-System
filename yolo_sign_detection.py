from ultralytics import YOLO
import cv2
import numpy as np

# Global cache
_models = {}

def load_yolo(model_name="best.pt"):
    global _models
    if model_name not in _models:
        print(f"Loading YOLO model: {model_name}...")
        _models[model_name] = YOLO(model_name)
        print(f"YOLO model {model_name} loaded.")
    return _models[model_name]

def detect_sign(image, conf_threshold=0.5, model_type='sign'):
    """
    Run YOLO detection.
    model_type: 'sign' (loads best.pt) or 'general' (loads yolo11n.pt)
    """
    try:
        # Select model
        if model_type == 'general':
            model_name = "yolo11n.pt" # Standard YOLOv11 Nano for general objects
        else:
            model_name = "best.pt" # Custom Sign Detection Model
            
        model = load_yolo(model_name)
        
        # Convert PIL to numpy (OpenCV format) if needed
        if not isinstance(image, np.ndarray):
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            frame = image
            
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        detections = results[0].boxes
        highest_conf = 0
        best_class = None
        
        for box in detections:
            conf = float(box.conf[0])
            if conf > highest_conf:
                highest_conf = conf
                class_index = int(box.cls[0])
                best_class = model.names[class_index]
                
        return best_class, annotated_frame
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None, None
