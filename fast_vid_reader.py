import cv2
import os


def video_reader(stream_id, uri, frame_queue, stop_event):
    """
    A dedicated process for reading frames from a single video stream.
    
    This process continuously reads frames from its assigned stream and
    places them onto a shared queue. By doing this in a separate process,
    the time consuming read call does not block other readers.
    
    Args:
        stream_id (int): A unique identifier for the video stream.
        uri (str): The URI of the video stream.
        frame_queue (Queue): The multiprocessing queue to push frames to.
        stop_event (Event): An event to signal the process to stop.
    """
    print(f"[{os.getpid()}] Starting video reader for stream {stream_id}...")
    
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        print(f"[{os.getpid()}] Error: Could not open video stream {stream_id} at {uri}")
        stop_event.set() # Signal main process to stop if a stream fails
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        
        if not ret:
            print(f"[{os.getpid()}] Reached end of stream {stream_id}. Exiting.")
            break
        
        # Hand off the frame immediately to the shared queue.
        # This is the key to decoupling the reader from the processor.
        # The reader does not wait for the frame to be processed.
        try:
            frame_queue.put((stream_id, frame), block=False)
        except Exception as e:
            # If the queue is full, the video reader will be blocked,
            # but this is a good indicator that the processors can't keep up.
            print(f"[{os.getpid()}] Queue is full, dropping frame from stream {stream_id}")
            continue

    cap.release()
    print(f"[{os.getpid()}] Video reader for stream {stream_id} has stopped.")