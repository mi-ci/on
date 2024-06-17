from ultralytics import YOLO
model = YOLO('ep100.pt')
model.export(format="tflite")
