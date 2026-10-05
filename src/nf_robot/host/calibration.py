import cv2
import numpy as np
import time
import glob
import argparse
import logging
import asyncio
import concurrent.futures
import json
import queue
import threading

import av
import websockets

from nf_robot.robot.spools import SpiralCalculator
from nf_robot.common.pose_functions import *
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
from nf_robot.generated.nf import config

logger = logging.getLogger(__name__)

#the number of squares on the board (width and height)
board_w = 14
board_h = 9
# side length of one square in meters
board_dim = 0.075

def collect_images_locally_raspi(num_images, resolution_str):
    """
    Collects images locally on a Raspberry Pi using the picamera2 library.
    
    Args:
        num_images (int): The number of images to collect.
        resolution_str (str): The resolution as a string, e.g., "4608x2592".
    """
    try:
        from picamera2 import Picamera2
        from libcamera import Transform, controls
    except ImportError:
        logging.error("picamera2 or libcamera not found. This function is for Raspberry Pi only.")
        return
        
    width, height = map(int, resolution_str.split('x'))

    picam2 = Picamera2()
    capture_config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.000001, "AfSpeed": controls.AfSpeedEnum.Fast}) 
    logging.info("Started Pi camera.")
    time.sleep(1)
    for i in range(num_images):
        time.sleep(1)
        im = picam2.capture_array()
        cv2.imwrite(f"images/cal/cap_{i}.jpg", im)
        time.sleep(1)
        logging.info(f'Collected ({i+1}/{num_images}) images.')

def collect_images_stream(address, num_images):
    """
    Connects to a video stream and collects a specified number of images.
    
    Args:
        address (str): The network address of the video stream.
        num_images (int): The number of images to collect.
    """
    logging.info(f'Connecting to {address}...')
    cap = cv2.VideoCapture(address)
    logging.debug(f'Video capture object: {cap}')
    last_cap_time = time.time()
    i = 0
    while i < num_images:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame from stream. Retrying...")
            return
        if time.time() > last_cap_time+1:
            fpath = f'images/cal2/cap_{i}.png'
            cv2.imwrite(fpath, frame)
            i += 1
            logging.info(f'Saved frame to {fpath}')
            last_cap_time = time.time()

def is_blurry(image, threshold=6.0):
    """
    Checks if an image is too blurry based on Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ensure grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def fit_intrinsics(opts, ipts, image_shape):
    """cv2.calibrateCamera with the principal point held at the image centre.
    Returns (rms reprojection error, intrinsic matrix, distortion, rvecs, tvecs)."""
    # Initialize the Matrix with the Image Center
    # This tells OpenCV: "Start assuming the lens is perfectly centered"
    w, h = image_shape
    intrinsic_matrix = np.array([
        [1000.0, 0.0,    w / 2.0], # f_x estimate, 0, c_x
        [0.0,    1000.0, h / 2.0], # 0, f_y estimate, c_y
        [0.0,    0.0,    1.0    ]
    ], dtype=np.float32)

    # Use Flags to Lock the Center
    # CALIB_USE_INTRINSIC_GUESS: Use the matrix above as the starting point
    # CALIB_FIX_PRINCIPAL_POINT: Do NOT move c_x and c_y during optimization
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT

    return cv2.calibrateCamera(opts, ipts, image_shape, intrinsic_matrix, None, flags=flags)

# calibrate interactively
class CalibrationInteractive:
    def __init__(self, config_file, board_w=board_w, board_h=board_h, board_dim=board_dim, cal_field='camera_cal', display=True):
        #Initializing variables
        self.board_w = board_w
        self.board_h = board_h
        self.cal_field = cal_field
        self.display = display
        board_n = board_w * board_h
        self.opts = []
        self.ipts = []
        self.intrinsic_matrix = np.zeros((3, 3), np.float32)
        self.distCoeffs = np.zeros((5, 1), np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        self.config_file = config_file

        # prepare object points based on the actual dimensions of the calibration board
        # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
        self.objp = np.zeros((board_n,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)
        self.objp = self.objp * board_dim
        logging.debug(f'Object points:\n{self.objp}')

        self.images_obtained = 0
        self.image_shape = None
        self.cnt = 0

    def addImage(self, image):
        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.image_shape = grey_image.shape[::-1]
        #Find chessboard corners
        logging.debug(f'Searching image {self.cnt}')
        self.cnt+=1
        found, corners = cv2.findChessboardCornersSB(grey_image, (self.board_w,self.board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        # found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ADAPTIVE_THRESH)

        if found == True:
            #Add the "true" checkerboard corners
            self.opts.append(self.objp)

            self.ipts.append(corners)
            self.images_obtained += 1 
            logging.info(f"Chessboards obtained: {self.images_obtained}")

            image = cv2.drawChessboardCorners(image, (self.board_w,self.board_h), corners, found)
        # this resize is only for display and should not affect calibration
        if self.display:
            image = cv2.resize(image, (1920, 1080),  interpolation = cv2.INTER_LINEAR)
            try:
                cv2.imshow('img', image)
                cv2.waitKey(500)
            except cv2.error as e:
                # headless opencv builds have no GUI support; keep calibrating without preview
                logging.warning(f'Disabling image preview (no GUI support in this OpenCV build): {e}')
                self.display = False

    def calibrate(self):
        if self.images_obtained < 20:
            logging.error(f'Obtained {self.images_obtained} images of checkerboard. Required 20.')
            raise RuntimeError(f'Obtained {self.images_obtained} images of checkerboard. Required 20')

        logging.info('Running calibrations...')
        # ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(
        #     self.opts, self.ipts, self.image_shape, None, None)

        ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = fit_intrinsics(
            self.opts, self.ipts, self.image_shape)

        #Save matrices
        logging.info(f"Camera calibration performed with image resolution: {self.image_shape[0]}x{self.image_shape[1]}.")
        logging.info('Intrinsic Matrix:')
        logging.info(str(self.intrinsic_matrix))
        logging.info('Distortion Coefficients:')
        logging.info(str(self.distCoeff))
        logging.info('Calibration complete.')

        #Calculate the total reprojection error.  The closer to zero the better.
        tot_error = 0
        for i in range(len(self.opts)):
            imgpoints2, _ = cv2.projectPoints(self.opts[i], rvecs[i], tvecs[i], self.intrinsic_matrix, self.distCoeff)
            error = cv2.norm(self.ipts[i].reshape(-1, 2), imgpoints2.reshape(-1, 2), cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        terr = tot_error/len(self.opts)
        logging.info(f"Total reprojection error: {terr}")

    def save(self):
        logging.info(f'Saving data to {self.config_file} field "{self.cal_field}"...')
        cfg = load_config(path=self.config_file)
        cal = getattr(cfg, self.cal_field)
        cal.intrinsic_matrix = self.intrinsic_matrix.flatten().tolist()
        cal.distortion_coeff = self.distCoeff.flatten().tolist()
        cal.resolution.width = self.image_shape[0]
        cal.resolution.height = self.image_shape[1]
        save_config(cfg, self.config_file)

# calibrate from files locally
def calibrate_from_files(config_file, image_dir='images/cap', board_w=board_w, board_h=board_h, board_dim=board_dim, cal_field='camera_cal'):
    ce = CalibrationInteractive(config_file, board_w=board_w, board_h=board_h, board_dim=board_dim, cal_field=cal_field)
    filepaths = glob.glob(f'{image_dir}/*.jpg') + glob.glob(f'{image_dir}/*.png')
    if not filepaths:
        raise RuntimeError(f'No .jpg or .png images found in {image_dir}')
    for filepath in filepaths:
        logging.info(f"Analyzing {filepath}")
        image = cv2.imread(filepath)
        ce.addImage(image)
    ce.calibrate()
    ce.save()

def calibrate_from_stream(address, config_file):
    logging.info(f'Connecting to {address}...')
    cap = cv2.VideoCapture(address)
    logging.debug(f'Video capture object: {cap}')
    ce = CalibrationInteractive(config_file)
    i=0
    while ce.images_obtained < 20:
        ret, frame = cap.read()
        if ret and i%10==0:
            ce.addImage(frame)
        i+=1
    ce.calibrate()
    ce.save()

class ComponentVideoSession:
    """A websocket client of a component server, held open only so the server streams
    video. Runs its own event loop in a thread and hands each video_ready port to the
    caller through a queue. The server kills rpicam-vid when this disconnects."""
    def __init__(self, address, ws_port):
        self.uri = f'ws://{address}:{ws_port}'
        self.video_ports = queue.Queue()
        self.loop = None
        self.task = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        self.task = self.loop.create_task(self._session())
        try:
            self.loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            pass
        finally:
            self.loop.close()

    async def _session(self):
        async with websockets.connect(self.uri, max_size=None, open_timeout=10) as ws:
            logging.info(f'Connected to {self.uri}, waiting for video_ready')
            # measurements arrive many times a second and are drained here unread
            async for message in ws:
                update = json.loads(message)
                if 'video_ready' in update:
                    self.video_ports.put(int(update['video_ready'][0]))

    def stop(self):
        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.task.cancel)
        self.thread.join(timeout=5)


class LatestFrameReader(threading.Thread):
    """Decodes a video stream as fast as it arrives, keeping only the newest frame, so a
    consumer slower than the stream sees the present rather than a growing backlog."""
    OPEN_ATTEMPTS = 5
    OPEN_RETRY_S = 1.5

    def __init__(self, uri):
        super().__init__(daemon=True)
        self.uri = uri
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_seq = 0

    def run(self):
        options = {'fflags': 'nobuffer', 'flags': 'low_delay', 'fast': '1'}
        container = None
        try:
            # the socket may not be accepting yet when video_ready arrives
            for attempt in range(self.OPEN_ATTEMPTS):
                try:
                    container = av.open(self.uri, options=options, mode='r')
                    break
                except (av.error.ConnectionRefusedError, av.error.TimeoutError):
                    if attempt == self.OPEN_ATTEMPTS - 1:
                        raise
                    time.sleep(self.OPEN_RETRY_S)
            logging.info(f'Receiving video from {self.uri}')
            for frame in container.decode(video=0):
                if self.stop_event.is_set():
                    break
                image = frame.to_ndarray(format='bgr24')
                with self.lock:
                    self.frame = image
                    self.frame_seq += 1
        except av.error.FFmpegError as e:
            logging.warning(f'Video stream ended: {e}')
        finally:
            if container is not None:
                container.close()

    def latest(self):
        """(sequence number, frame); the number changes only when a new frame arrives."""
        with self.lock:
            return self.frame_seq, self.frame

    def stop(self):
        self.stop_event.set()
        self.join(timeout=3)


def format_calibration_source(K, dist, image_shape):
    """The fit as the camera_cal_wide block of create_default_config, ready to paste."""
    w, h = image_shape
    return (
        f"    config.camera_cal_wide.resolution = nf_config.Resolution(width={w}, height={h})\n"
        f"    intrinsic_np = np.array([\n"
        f"        [{K[0, 0]:.4f},   0.,       {K[0, 2]:.1f}],\n"
        f"        [  0.,       {K[1, 1]:.4f}, {K[1, 2]:.1f}],\n"
        f"        [  0.,         0.,         1.]\n"
        f"    ])\n"
        f"    config.camera_cal_wide.intrinsic_matrix = intrinsic_np.flatten().tolist()\n"
        f"    distortion_np = np.array([{', '.join(f'{d:.8f}' for d in np.ravel(dist))}])\n"
        f"    config.camera_cal_wide.distortion_coeff = distortion_np.flatten().tolist()")


def calibrate_continuous(address, ws_port=8765, board_w=board_w, board_h=board_h,
                         board_dim=board_dim, min_motion_px=10.0):
    """Calibrate a component's camera from its live stream until stopped with q, Esc or
    Ctrl-C, refitting in the background as boards are collected.

    Connects as an ordinary client, so the component streams in its default mode: the
    one the robot runs in, which is the one worth calibrating. Nothing else may be
    connected to the component, or the two clients' rpicam-vid launches fight over the
    camera.

    A board is collected when its corners have moved at least min_motion_px on average
    since the last one collected, so holding the board still does not pile up copies of
    one view and outweigh the rest.
    """
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * board_dim

    ipts = []
    image_shape = None
    best = None  # (rms, K, dist, number of boards fitted)
    fitter = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fit_future = None

    def harvest(future):
        nonlocal best
        rms, K, dist, _, _ = future.result()
        best = (rms, K, dist, future.n_boards)
        logging.info(f'{future.n_boards} boards: fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} '
                     f'rms={rms:.3f}px dist={np.round(np.ravel(dist), 4).tolist()}')

    session = ComponentVideoSession(address, ws_port)
    session.start()
    reader = None
    window = 'calibration'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        last_seq = 0
        while True:
            if reader is None or not reader.is_alive():
                # wait out a missing or ended stream until the component announces one
                try:
                    port = session.video_ports.get(timeout=0.1)
                except queue.Empty:
                    if not session.thread.is_alive():
                        raise RuntimeError(f'Lost the websocket connection to {session.uri}')
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    continue
                reader = LatestFrameReader(f'tcp://{address}:{port}')
                reader.start()

            seq, frame = reader.latest()
            if frame is None or seq == last_seq:
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                continue
            last_seq = seq

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_shape = grey.shape[::-1]
            found, corners = cv2.findChessboardCornersSB(
                grey, (board_w, board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
            if found:
                corners = corners.reshape(-1, 1, 2)
                novel = not ipts or np.linalg.norm(corners - ipts[-1], axis=2).mean() >= min_motion_px
                if novel:
                    ipts.append(corners)
                cv2.drawChessboardCorners(frame, (board_w, board_h), corners, found)

            if fit_future is not None and fit_future.done():
                harvest(fit_future)
                fit_future = None
            # a fit needs a handful of views to be determined at all
            if fit_future is None and len(ipts) >= 5 and (best is None or best[3] < len(ipts)):
                fit_future = fitter.submit(fit_intrinsics, [objp] * len(ipts), list(ipts), image_shape)
                fit_future.n_boards = len(ipts)

            status = f'boards {len(ipts)}'
            if best is not None:
                status += f'  fx {best[1][0, 0]:.1f}  fy {best[1][1, 1]:.1f}  rms {best[0]:.3f}px ({best[3]})'
            cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.stop()
        session.stop()
        cv2.destroyAllWindows()
        fitter.shutdown(wait=True)

    if len(ipts) < 5:
        logging.error(f'Only {len(ipts)} boards collected; not enough to fit')
        return
    # the background fit may have stopped short of the last boards collected
    logging.info(f'Final fit over all {len(ipts)} boards...')
    rms, K, dist, _, _ = fit_intrinsics([objp] * len(ipts), ipts, image_shape)
    logging.info(f'Final: {len(ipts)} boards, fx={K[0, 0]:.4f} fy={K[1, 1]:.4f} rms={rms:.4f}px')
    print(format_calibration_source(K, dist, image_shape))


def main():
    parser = argparse.ArgumentParser(description='Run robot calibration functions. Use --help for more details on each command.')
    parser.add_argument('--mode', type=str, choices=[
        'collect-images-stream',
        'calibrate-from-files',
        'collect-images-locally-raspi',
        'calibrate-from-stream',
        'continuous'
    ], required=True, help='Choose the calibration function to run:\n \
            "collect-images-stream" to capture a specified number of images from a network stream; \
            "calibrate-from-files" to run camera calibration on a local set of images; \
            "collect-images-locally-raspi" to capture a specified number of images from a connected camera on a Raspberry Pi; \
            "calibrate-from-stream" to run camera calibration directly from a network stream until 20 images are collected; \
            "continuous" to connect to the component at --pi as a client, collect boards from its default stream until stopped, and print the fit as source.')
    parser.add_argument('--pi', type=str,
                        help='IP address of the component to calibrate (used with "continuous").')
    parser.add_argument('--ws-port', type=int, default=8765,
                        help='Websocket port of the component server (used with "continuous").')
    parser.add_argument('--address', type=str, default='tcp://192.168.1.151:8888',
                        help='The network address for the video stream (used with stream modes).')
    parser.add_argument('--num-images', type=int, default=50,
                        help='The number of images to collect when using "collect-images-locally-raspi" or "collect-images-stream".')
    parser.add_argument('--resolution', type=str, default='4608x2592',
                        help='The resolution for the camera on the Raspberry Pi (e.g., "4608x2592"). Used with "collect-images-locally-raspi" mode.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path of the config file to write/update with calibrated values.')
    parser.add_argument('--image-dir', type=str, default='images/cap',
                        help='Directory of .jpg/.png images to calibrate from (used with "calibrate-from-files").')
    parser.add_argument('--board-width', type=int, default=board_w,
                        help='Number of inner corners along the board width.')
    parser.add_argument('--board-height', type=int, default=board_h,
                        help='Number of inner corners along the board height.')
    parser.add_argument('--square-size', type=float, default=board_dim * 1000.0,
                        help='Side length of a single checkerboard square, in millimeters.')
    parser.add_argument('--wide', action='store_true',
                        help='Save the result to the config\'s wide camera calibration field (camera_cal_wide) instead of camera_cal.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cal_field = 'camera_cal_wide' if args.wide else 'camera_cal'

    if args.mode == 'collect-images-locally-raspi':
        collect_images_locally_raspi(args.num_images, args.resolution)
    elif args.mode == 'collect-images-stream':
        collect_images_stream(args.address, args.num_images)
    elif args.mode == 'calibrate-from-files':
        calibrate_from_files(
            args.config,
            image_dir=args.image_dir,
            board_w=args.board_width,
            board_h=args.board_height,
            board_dim=args.square_size / 1000.0,
            cal_field=cal_field,
        )
    elif args.mode == 'calibrate-from-stream':
        calibrate_from_stream(args.address, args.config)
    elif args.mode == 'continuous':
        if not args.pi:
            parser.error('--pi is required with --mode continuous')
        calibrate_continuous(
            args.pi,
            ws_port=args.ws_port,
            board_w=args.board_width,
            board_h=args.board_height,
            board_dim=args.square_size / 1000.0,
        )

if __name__ == "__main__":
    main()
