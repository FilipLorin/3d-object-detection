import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pypylon import pylon
import os
from datetime import datetime
from matplotlib import pyplot as plt


NUM_DISPARITIES = 512
BLOCK_SIZE = 65

def attachCameras():
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    cameras = [pylon.InstantCamera(tlf.CreateDevice(device)) for device in devices]
    return cameras

def aquireImg(path=None, cam=None):
    if path is not None:
        return cv.imread(path)
    elif cam is not None:
        cam.Open() #TODO: fix BNW
        cam.StartGrabbing(1)
        grab = cam.RetrieveResult(2000)
        print(grab.GrabSucceeded())
        img = grab.GetArray()
        cam.Close()
        return img
    else:
        raise Error("No cam object or image path provided.")

def cv_imshow(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def loadCalibrationData(calibration_file_path):
    calibration_data = np.load(calibration_file_path)
    return data['camMatrix'], data['dist_Coeff']

def applyCorrection(img, calibration_file_path):
    camMatrix, distCoeff = loadCalibrationData(calibration_file_path)
    h, w = img.shape[:2]
    newCamMatrix, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (w, h), 1, (w, h))   
    undistorted = cv.undistort(img, camMatrix, distCoeff, None, newCamMatrix)
    return undistorted

def getDisparity(left_image, right_image):
    #print(np.shape(left_image))
    left_grayscale  = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY);
    right_grayscale = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY);
    stereo = cv.StereoSGBM_create(
        minDisparity=1, 
        numDisparities=NUM_DISPARITIES, 
        blockSize=BLOCK_SIZE, 
        #P1=8*BLOCK_SIZE*BLOCK_SIZE, 
        #P2=32*BLOCK_SIZE*BLOCK_SIZE,
        #uniquenessRatio=12
        )
    disparity = stereo.compute(left_grayscale, right_grayscale)
    return disparity

def getDepth(dispatity_map): # results in m
    focal_length = 6.5 #mm
    camera_baseline_dist = 0.180 #m
    pixel_size = 0.012 #mm
    m = focal_length * camera_baseline_dist / pixel_size
    return np.divide(m, np.clip(dispatity_map, a_max = None, a_min=1))  

def getObjects(image):
    model = YOLO("yolo11n.pt")  
    results = model(image)
    return results[0]

def getObjectsInfo(detection_result):
    boxes, classes = detection_result.boxes, detection_result.boxes.cls
    objects = [
        {
            'class': detection_result.names[int(cl)],
            'xyxy': boxes.xyxy[i, :]
        }
        for i, cl in enumerate(classes)
    ]
    return objects

def getObjectDistance(obj, depth_map):
    x1, y1, x2, y2 = [int(x) for x in obj['xyxy']]
    depth_data = depth_map[x1:x2, y1:y2] #TODO correct for coordinate systems
    distance = np.mean(depth_data)
    return distance
    
def save_view(left_im, right_im):
    idd = str(datetime.now()).split('.')[1]
    cv.imwrite(f"{idd}_left.png", left_im)
    cv.imwrite(f"{idd}_right.png", right_im)

def resize(img, scale_factor):
    w, h = np.shape(img)
    return cv.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv.INTER_LINEAR)
    
def normalize(img):
    dmax = img.max()
    dmin = img.min()
    return np.uint8((255*img-dmin)/(dmax-dmin))


if __name__ == "__main__":
    # test object detection
    """
    cameras = attachCameras()
    img = aquireImg(cam=cameras[0])
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    results = getObjects(img) 
    results.show()
    """

    # test depth detection
    img_l = aquireImg(path="014664_left.png")
    img_r = aquireImg(path="014664_right.png")
    #save_view(img_l, img_r)
    disparity = getDisparity(img_l, img_r)
    depth = getDepth(disparity)
    cv_imshow(normalize(resize(disparity, 0.12)))
    cv_imshow(normalize(resize(depth, 0.12)))
    #plt.hist(disparity.ravel(), 100, [0, 500]); plt.show()
    #plt.hist(depth.ravel(), 255, range=[0, 10]); plt.show()
    
    """
    for obj in getObjectsInfo(results):
    	print([obj['class'], getObjectDistance(obj, depth)])
    """