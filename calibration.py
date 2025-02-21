import cv2
import cv2.aruco as aruco
import numpy as np
from time import time, sleep
import glob

# when on the raspi, just collect the images. it doesn't have enough ram to analyze them.
def collect_images():
    from picamera2 import Picamera2
    from libcamera import Transform
    picam2 = Picamera2()
    capture_config = picam2.create_still_configuration(main={"size": (4608, 2592), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()
    print("started pi camera")
    for i in range(20):
        im = picam2.capture_array()
        cv2.imwrite(f"images/cal/cap_{i}.jpg", im)
        sleep(1)
        print(f'collected ({i+1}/20)')
    
def calibate_camera():
    #Input the number of board images to use for calibration (recommended: ~20)
    n_boards = 20
    #Input the number of squares on the board (width and height)
    board_w = 9
    board_h = 6
    # side length of one square in meters
    board_dim = 0.02246
    #Initializing variables
    board_n = board_w * board_h
    opts = []
    ipts = []
    npts = np.zeros((n_boards, 1), np.int32)
    intrinsic_matrix = np.zeros((3, 3), np.float32)
    distCoeffs = np.zeros((5, 1), np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # prepare object points based on the actual dimensions of the calibration board
    # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
    objp = np.zeros((board_h*board_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:(board_w*board_dim):board_dim,0:(board_h*board_dim):board_dim].T.reshape(-1,2)

    #Loop through the images.  Find checkerboard corners and save the data to ipts.
    images_obtained = 0
    for filepath in glob.glob('images/cal/*.jpg'):
        print(f"analyzing {filepath}")
        image = cv2.imread(filepath)

        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
 
        #Find chessboard corners
        #found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        found, corners = cv2.findChessboardCornersSB(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ACCURACY)

        if found == True:
            #Add the "true" checkerboard corners
            opts.append(objp)
            
            #Improve the accuracy of the checkerboard corners found in the image and save them to the ipts variable.
            # cv2.cornerSubPix(grey_image, corners, (20, 20), (-1, -1), criteria)

            ipts.append(corners)
            images_obtained += 1 
            print("chessboards obtained {}/{}".format(images_obtained, n_boards))
    
    #Calibrate the camera
    print('Running Calibrations...')
    ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, grey_image.shape[::-1],None,None)

    #Save matrices
    print('Intrinsic Matrix: ')
    print(str(intrinsic_matrix))
    print('Distortion Coefficients: ')
    print(str(distCoeff))

    #Save data
    print('Saving data file to calibration_data.npz')
    np.savez('calibration_data', distCoeff=distCoeff, intrinsic_matrix=intrinsic_matrix)
    print('Calibration complete')

    #Calculate the total reprojection error.  The closer to zero the better.
    tot_error = 0
    for i in range(len(opts)):
        imgpoints2, _ = cv2.projectPoints(opts[i], rvecs[i], tvecs[i], intrinsic_matrix, distCoeff)
        error = cv2.norm(ipts[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    print("Total reprojection error: ", tot_error/len(opts))
    return True

if __name__ == "__main__":
    calibate_camera()
