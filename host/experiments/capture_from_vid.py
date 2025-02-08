import cv2

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
        cv2.imshow("Received Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    else:
        print("Could not retrieve frame")

cv2.destroyAllWindows()