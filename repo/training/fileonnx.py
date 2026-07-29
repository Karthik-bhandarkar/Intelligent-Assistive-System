import torch
from ultralytics import YOLO

# Step 1: Load the YOLO model
model = YOLO("best.pt")

# Step 2: Export the model to ONNX format
model.export(format='onnx')

print("Model exported to ONNX format successfully!")


#pip install ultralytics onnx onnxruntime
