# INPUT: image
# OUTPUT: [box, category] list

# Requirements: ultralytics, pip 
import cv2
from ultralytics import YOLO

# Download and load in pretrained COCO detection model
model = YOLO("yolo11n.pt")

im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

