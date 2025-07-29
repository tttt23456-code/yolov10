from ultralytics import YOLOv10

# Load a model
model = YOLOv10('yolov10s.pt')  # load an official model
model = YOLOv10('runs/detect/train6/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')