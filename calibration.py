import numpy as np
import cv2 as cv
import glob
import os

def calibrate(nRows=7, nCols=11):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'calibration images')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg')) + glob.glob(os.path.join(calibrationDir, '*.png'))
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)
    
    worldPtsList = []
    imgPtsList = []
    
    success_count = 0
    total_images = len(imgPathList)
    imgGray = None  
    
    for imgPath in imgPathList:
        imgBGR = cv.imread(imgPath)
        if imgBGR is None:
            continue
            
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        
        flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), flags)
        
        if cornersFound:
            cornersSubPix = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            worldPtsList.append(worldPtsCur)
            imgPtsList.append(cornersSubPix)
            success_count += 1
            
    print(f"processed {success_count} out of {total_images} images")
    
    if success_count == 0:
        return None, None
    
    if imgGray is None:
        return None, None
    
    ret, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    
    reprojError = 0
    for i in range(len(worldPtsList)):
        imgPts2, _ = cv.projectPoints(worldPtsList[i], rvecs[i], tvecs[i], camMatrix, distCoeff)
        error = cv.norm(imgPtsList[i], imgPts2, cv.NORM_L2) / len(imgPts2)
        reprojError += error
    reprojError /= len(worldPtsList)
    
    paramPath = os.path.join(root, 'camera_calibration.npz')
    np.savez(paramPath, camMatrix=camMatrix, distCoeff=distCoeff)
    
    return camMatrix, distCoeff

if __name__ == '__main__':
    calibrate()
