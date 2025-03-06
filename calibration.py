import cv2
import cv2.aruco as aruco
import numpy as np
from time import time, sleep
import glob

#the number of squares on the board (width and height)
board_w = 14
board_h = 9
# side length of one square in meters
board_dim = 0.021

# when on the raspi, just collect the images. it doesn't have enough ram to analyze them.
def collect_images():
    from picamera2 import Picamera2
    from libcamera import Transform, controls
    picam2 = Picamera2()
    capture_config = picam2.create_still_configuration(main={"size": (4608, 2592), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.000001, "AfSpeed": controls.AfSpeedEnum.Fast}) 
    print("started pi camera")
    sleep(1)
    for i in range(50):
        sleep(1)
        im = picam2.capture_array()
        cv2.imwrite(f"images/cal/cap_{i}.jpg", im)
        sleep(1)
        print(f'collected ({i+1}/20)')

def collect_images_stream():
    video_uri = 'tcp://192.168.1.153:8888'
    print(f'Connecting to {video_uri}')
    cap = cv2.VideoCapture(video_uri)
    print(cap)
    i = 0
    while i < 50:
        ret, frame = cap.read()
        if not ret:
            continue
        fpath = f'images/cal/cap_{i}.jpg'
        cv2.imwrite(fpath, frame)
        i += 1
        print(f'saved frame to {fpath}')
        sleep(1)

def is_blurry(image, threshold=6.0):
    """
    Checks if an image is too blurry based on Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ensure grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# calibrate interactively
class CalibrationInteractive:
    def __init__(self):
        #Initializing variables
        board_n = board_w * board_h
        self.opts = []
        self.ipts = []
        self.intrinsic_matrix = np.zeros((3, 3), np.float32)
        self.distCoeffs = np.zeros((5, 1), np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        # prepare object points based on the actual dimensions of the calibration board
        # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
        self.objp = np.zeros((board_n,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:board_h,0:board_w].T.reshape(-1,2)
        self.objp = self.objp * board_dim

        self.images_obtained = 0
        self.image_shape = None
        self.cnt = 0

    def addImage(self, image):
        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.image_shape = grey_image.shape[::-1]
        #Find chessboard corners
        print(f'search image {self.cnt}')
        self.cnt+=1
        found, corners = cv2.findChessboardCornersSB(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ACCURACY)

        if found == True:
            #Add the "true" checkerboard corners
            self.opts.append(self.objp)

            self.ipts.append(corners)
            self.images_obtained += 1 
            print(f"chessboards obtained {self.images_obtained}")

            image = cv2.drawChessboardCorners(image, (14,9), corners, found)
        # this resize is only for display and should not affect calibration
        image = cv2.resize(image, (2304, 1296),  interpolation = cv2.INTER_LINEAR)
        cv2.imshow('img', image)
        cv2.waitKey(500)

    def calibrate(self):
        if self.images_obtained < 20:
            raise RuntimeError(f'Obtained {self.images_obtained} images of checkerboard. Required 20')

        print('Running Calibrations...')
        ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(
            self.opts, self.ipts, self.image_shape, None, None)

        #Save matrices
        print('Intrinsic Matrix: ')
        print(str(self.intrinsic_matrix))
        print('Distortion Coefficients: ')
        print(str(self.distCoeff))
        print('Calibration complete')

        #Calculate the total reprojection error.  The closer to zero the better.
        tot_error = 0
        for i in range(len(self.opts)):
            imgpoints2, _ = cv2.projectPoints(self.opts[i], rvecs[i], tvecs[i], self.intrinsic_matrix, self.distCoeff)
            error = cv2.norm(self.ipts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        terr = tot_error/len(self.opts)
        print("Total reprojection error: ", terr)

    def save(self): 
        fname = 'camera_coef.npz'
        print(f'Saving data file to {fname}')
        np.savez(fname, distCoeff=self.distCoeff, intrinsic_matrix=self.intrinsic_matrix)

# calibrate from files locally
def calibrate_from_files():
    ce = CalibrationInteractive()
    for filepath in glob.glob('images/cal/*.jpg'):
        print(f"analyzing {filepath}")
        image = cv2.imread(filepath)
        ce.addImage(image)
    ce.calibrate()
    ce.save()

def calibrate_from_stream():
    video_uri = 'tcp://192.168.1.151:8888'
    print(f'Connecting to {video_uri}')
    cap = cv2.VideoCapture(video_uri)
    print(cap)
    ce = CalibrationInteractive()
    i=0
    while ce.images_obtained < 20:
        ret, frame = cap.read()
        if ret and i%10==0:
            ce.addImage(frame)
        i+=1
    ce.calibrate()
    ce.save()

if __name__ == "__main__":
    calibrate_from_stream()
    # collect_images_stream()
    # calibrate_from_files()
