import cv2
import cv2.aruco as aruco
import numpy as np
from time import time

    
def calibate_camera():
    #Input the number of board images to use for calibration (recommended: ~20)
    n_boards = 20
    #Input the number of squares on the board (width and height)
    board_w = 10
    board_h = 7
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

    # for use on raspi. if on some other platform, change to some other method
    picam2 = Picamera2()
    capture_config = picam2.create_preview_configuration(main={"size": (2304, 1296), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()

    #Loop through the images.  Find checkerboard corners and save the data to ipts.
    images_obtained = 0
    start_time = time()
    timeout = 60
    while images_obtained < n_boards and time() < (start_time + timeout):
        #Capture image
        image = im = picam2.capture_array()

        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #Find chessboard corners
        found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found == True:
            #Add the "true" checkerboard corners
            opts.append(objp)

            #Improve the accuracy of the checkerboard corners found in the image and save them to the ipts variable.
            cv2.cornerSubPix(grey_image, corners, (20, 20), (-1, -1), criteria)
            ipts.append(corners)
            images_obtained += 1 
            print("images obtained {}/{}".format(images_obtained, n_boards))
    
    if images_obtained < n_boards:
        print("Timed out before obtaining enough images of the calibration board")
        return False
    
    print('Finished capturing images.')

    #Calibrate the camera
    print('Running Calibrations...')
    ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, grey_image.shape[::-1],None,None)

    #Save matrices
    print('Intrinsic Matrix: ')
    print(str(intrinsic_matrix))
    print('Distortion Coefficients: ')
    print(str(distCoeff))

    #Save data
    print('Saving data file to camera_calibration.npz')
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