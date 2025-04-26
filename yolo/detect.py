#! /usr/bin/python3
# INPUT: image
# OUTPUT: python dict with detected objects as string in stdout

# Requirements: python with installed: ultralytics, opencv-python 
import cv2
from ultralytics import YOLO

# Download and load in pretrained COCO detection model
model = YOLO("yolo11n.pt")

im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=False)
print(results[0].boxes)
