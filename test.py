from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

model = YOLO("runs/detect/train/weights/last.pt")

# Train the model
#results = model.train(resume=True, epochs=15, imgsz=640, device="cpu")
model.train(resume=False, epochs=1, imgsz=640, device="cpu")
