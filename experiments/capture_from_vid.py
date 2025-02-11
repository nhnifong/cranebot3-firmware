import cv2
import numpy as np
from cv_common import locate_markers

# Intrinsic Matrix: 
camera_matrix = np.array(
[[1.55802968e+03, 0.00000000e+00, 8.58167917e+02],
 [0.00000000e+00, 1.56026885e+03, 6.28095370e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
 
# Distortion Coefficients: 
dist_coeffs = np.array(
[[ 3.40916628e-01, -2.38650897e+00, -8.85125582e-04, 3.34240054e-03, 4.69525036e+00]])

def receive_video_stream(stream_url):
    """Receives and decodes a video stream using cv2.VideoCapture.

    Args:
        stream_url: The URL of the video stream (e.g., "tcp://<ip>:<port>").

    Returns:
        A generator that yields frames (NumPy arrays) or None on error.
    """
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error opening video stream: {stream_url}")
        yield None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of stream or error

        yield frame

    cap.release()

# Example usage:
raspberry_pi_ip = "192.168.1.151"  # Replace with your Pi's IP
port_number = 8888  # Replace with your port
stream_url = f"tcp://{raspberry_pi_ip}:{port_number}"  # Construct the URL

for frame in receive_video_stream(stream_url):
    if frame is not None:

        detections = locate_markers(frame)
        if len(detections) > 0:
            print(f"Found markers: {detections}")
            for d in detections:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, d.rotation, d.translation, 0.015)

        cv2.imshow("Received Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    else:
        print("Could not retrieve frame")

cv2.destroyAllWindows()