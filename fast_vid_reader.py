import os
import av
import time

options = {
    'rtsp_transport': 'tcp',
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'fast': '1',
}

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
    try:
        container = av.open(video_uri, options=options, mode='r')
        stream = next(s for s in container.streams if s.type == 'video')
        stream.thread_type = "SLICE"
        frame_queue.put(True) # TODO, what does it look like when connection does not succeed
        fnum = -1
        for av_frame in container.decode(stream):
            if stop_event.is_set():
                break
            frame = av_frame.to_ndarray(format='bgr24')
            fnum += 1
            # Hand off the frame immediately to the queue
            frame_queue.put((fnum,frame), block=False)
    finally:
        if 'container' in locals():
            container.close()

    print(f"[{os.getpid()}] Video reader for stream {stream_id} has stopped.")