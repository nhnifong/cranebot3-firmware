#Cam_calib_web.py
#Created by Chris Rillahan
#Last Updated: 02/03/2015
#Written with Python 2.7.2, OpenCV 2.4.8 and NumPy 1.8.0

#This program calculates the distortion parameters of a GoPro camera.
#A video must first be taken of a chessboard pattern moved to a variety of positions
#in the field of view with a GoPro.  

import cv2, sys
import numpy as np
import urllib.request
from time import time

#Import Information
filename = 'GOPR3697.MP4'
#Input the number of board images to use for calibration (recommended: ~20)
n_boards = 20
#Input the number of squares on the board (width and height)
board_w = 9
board_h = 6
#Board dimensions (typically in cm)
board_dim = 23.81
#Image resolution
image_size = (1920, 1080)

#The ImageCollect function requires two input parameters.  Filename is the name of the file
#in which checkerboard images will be collected from.  n_boards is the number of images of
#the checkerboard which are needed.  In the current writing of this function an additional 5
#images will be taken.  This ensures that the processing step has the correct number of images
#and can skip an image if the program has problems.

#This function loads the video file into a data space called video.  It then collects various
#meta-data about the file for later inputs.  The function then enters a loop in which it loops
#through each image, displays the image and waits for a fixed amount of time before displaying
#the next image.  The playback speed can be adjusted in the waitKey command.  During the loop
#checkerboard images can be collected by pressing the spacebar.  Each image will be saved as a
#*.png into the directory which stores this file.  The ESC key will terminate the function.
#The function will end once the correct number of images are collected or the video ends.
#For the processing step, try to collect all the images before the video ends.

def ImageCollect(filename, n_boards):
    #Collect Calibration Images
    print('-----------------------------------------------------------------')
    print('Loading video...')

    #Load the file given to the function
    video = cv2.VideoCapture(filename)
    #Checks to see if a the video was properly imported
    status = video.isOpened()

    if status == True:
        
        #Collect metadata about the file.
        FPS = video.get(cv2.CV_CAP_PROP_FPS)
        FrameDuration = 1/(FPS/1000)
        width = video.get(cv2.CV_CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))
        total_frames = video.get(cv2.CV_CAP_PROP_FRAME_COUNT)

        #Initializes the frame counter and collected_image counter
        current_frame = 0
        collected_images = 0

        #Video loop.  Press spacebar to collect images.  ESC terminates the function.
        while current_frame < total_frames:
            success, image = video.read()
            current_frame = video.get(cv2.CV_CAP_PROP_POS_FRAMES)
            cv2.imshow('Video', image)
            k = cv2.waitKey(int(FrameDuration)) #You can change the playback speed here FrameDuration
            if collected_images == n_boards: 
                break
            if k == 32:
                collected_images += 1
                cv2.imwrite('Calibration_Image' + str(collected_images) + '.png', image)
                print(str(collected_images) + ' images collected.')
            if k == 27:
                break
    
        #Clean up
        video.release()
        cv2.destroyAllWindows()
    else:
        print('Error: Could not load video')
        sys.exit()


#The ImageProcessing function performs the calibration of the camera based on the images
#collected during ImageCollect function.  This function will look for the images in the folder
#which contains this file.  The function inputs are the number of boards which will be used for
#calibration (n_boards), the number of squares on the checkerboard (board_w, board_h) as
#determined by the inside points (i.e. where the black squares touch).  board_dim is the actual
#size of the square, this should be an integer.  It is assumed that the checkerboard squares are
#square.

#This function first initializes a series of variables. Opts will store the true object points
#(i.e. checkerboard points).  Ipts will store the points as determined by the calibration images.
#The function then loops through each image.  Each image is converted to grayscale, and the
#checkerboard corners are located.  If it is successful at finding the correct number of corners
#then the true points and the measured points are stored into opts and ipts, respectively. The
#image with the checkerboard points are then displays.  If the points are not found that image
#is skipped.  Once the desired number of checkerboard points are acquired the calibration
#parameters (intrinsic matrix and distortion coefficients) are calculated.

#The distortion parameter are saved into a numpy file (calibration_data.npz).  The total
#total reprojection error is calculated by comparing the "true" checkerboard points to the
#image measured points once the image is undistorted.  The total reprojection error should be
#close to zero.

#Finally the function will go through the calbration images and display the undistorted image.
    
def ImageProcessing(n_boards, board_w, board_h, board_dim):
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
    for i in range(1, n_boards + 1):
    
        #Loading images
        # name = 'calibration/GOPR' + str(i+3701) + '.JPG'
        # print('Loading... ' + name)
        # image = cv2.imread(name)

        capture_url = "http://192.168.1.146/capture?_cb={}"
        req = urllib.request.urlopen(capture_url.format(time()*1000))
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)

    
        #Converting to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #Find chessboard corners
        found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(found)

        if found == True:

            #Add the "true" checkerboard corners
            opts.append(objp)

            #Improve the accuracy of the checkerboard corners found in the image and save them to the ipts variable.
            cv2.cornerSubPix(grey_image, corners, (20, 20), (-1, -1), criteria)
            ipts.append(corners)

            #Draw chessboard corners
            cv2.drawChessboardCorners(image, (board_w, board_h), corners, found)
        
            #Show the image with the chessboard corners overlaid.
            cv2.imshow("Corners", image)

        char = cv2.waitKey(0)

    cv2.destroyWindow("Corners") 
    
    print('')
    print('Finished processes images.')

    #Calibrate the camera
    print('Running Calibrations...')
    print(' ')
    ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, grey_image.shape[::-1],None,None)

    #Save matrices
    print('Intrinsic Matrix: ')
    print(str(intrinsic_matrix))
    print(' ')
    print('Distortion Coefficients: ')
    print(str(distCoeff))
    print(' ') 

    #Save data
    print('Saving data file...')
    np.savez('calibration_data', distCoeff=distCoeff, intrinsic_matrix=intrinsic_matrix)
    print('Calibration complete')

    #Calculate the total reprojection error.  The closer to zero the better.
    tot_error = 0
    for i in range(len(opts)):
        imgpoints2, _ = cv2.projectPoints(opts[i], rvecs[i], tvecs[i], intrinsic_matrix, distCoeff)
        error = cv2.norm(ipts[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    print("total reprojection error: ", tot_error/len(opts))

    #Undistort Images
    # for i in range(1, n_boards + 1):
    
    #     #Loading images
    #     print('Loading... Calibration_Image' + str(i) + '.png' )
    #     image = cv2.imread('Calibration_Image' + str(i) + '.png')

    #     # undistort
    #     dst = cv2.undistort(image.copy(), intrinsic_matrix, distCoeff, None)

    #     cv2.imshow('Undisorted Image',dst)

    #     char = cv2.waitKey(0)

    cv2.destroyAllWindows()


print("Starting camera calibration....")
print("Step 1: Image Collection")
print("We will playback the calibration video.  Press the spacebar to save")
print("calibration images.")
print(" ")
print('We will collect ' + str(n_boards) + ' calibration images.')

#ImageCollect(filename, n_boards)

print(' ')
print('All the calibration images are collected.')
print('------------------------------------------------------------------------')
print('Step 2: Calibration')
print('We will analyze the images take and calibrate the camera.')
print('Press the esc button to close the image windows as they appear.')
print(' ')

ImageProcessing(n_boards, board_w, board_h, board_dim)
