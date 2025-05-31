import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pypylon import pylon

def addCamera(ip):
    tlf = pylon.TlFactory.GetInstance()
    tl = tlf.CreateTl('BaslerGigE')
    cam_info = tl.CreateDeviceInfo()
    cam_info.SetIpAddress(ip)
    camera = pylon.InstantCamera(tlf.CreateDevice(cam_info))
    return camera

def aquireImg(path=None, cam=None):
    if path is not None:
        return cv.imread(path)
    elif cam is not None:
    	cam.Open()
    	cam.AcquisitionMode.SetValue('Continuous')
    	cam.TriggerMode.SetValue('Off')
    	cam.StartGrabbing(1)
    	grab = cam.RetrieveResult(2000)
    	print(grab.GrabSucceeded())
    	img = grab.GetArray()
    	cam.close()
    	return img
    else:
        raise Error("Not implemented")

def cv_imshow(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def loadCalibrationData(calibration_file_path):
    calibration_data = np.load(calibration_file_path)
    return data['camMatrix'], data['dist_Coeff']

def applyCalibratedCorrection():
    camMatrix, distCoeff = loadCalibrationData("camera_calibration.npz")
    h, w = img.shape[:2]
    newCamMatrix, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (w, h), 1, (w, h))   
    undistorted = cv.undistort(img, camMatrix, distCoeff, None, newCamMatrix)
    return undistorted

def getDisparity(left_image, right_image):
    left_grayscale  = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY);
    right_grayscale = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY);
    stereo = cv.StereoBM_create(numDisparities=128, blockSize=15) #TODO: Skalibrowac
    disparity = stereo.compute(left_grayscale, right_grayscale)
    return disparity

def getDepth(dispatity_map):
    focal_length = 12 #mm
    baseline_camera_distance = 10 #mm
    m = focal_length * baseline_camera_distance
    return m / dispatity_map  

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
    distance = np.min(depth_data)
    return distance


if __name__ == "__main__":
    cam1 = addCamera("192.168.1.20")
    # test object detection
    img = aquireImg(cam=cam1)
    results = getObjects(img)
    results.show()
    print(getObjectsInfo(results))
    # test depth detection
    img_l = aquireImg(path="left.png")
    img_r = aquireImg(path="right.png")
    disparity = getDisparity(img_l, img_r)
    depth = getDepth(disparity)
    cv_imshow(depth)


