# INPUT: image
# OUTPUT: [box, category] list

# Requirements: ultralytics, pip 

from ultralytics import YOLO

# Download and load in pretrained COCO detection model
model = YOLO("yolo11n.pt")
