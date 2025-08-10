from ultralytics import YOLO
import cv2
from PIL import Image

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="fsoco_test.yaml", epochs=300, imgsz=640, project="runs/detect")


im1=Image.open("/home/alessiovalle/gitRepos/cones_exam/amz/images/amz_00000.jpeg")
im2=cv2.imread("/home/alessiovalle/gitRepos/cones_exam/amz/images/amz_00000.jpeg")
valuations=model.val()
model.predict(source=[im1, im2], save=True)
