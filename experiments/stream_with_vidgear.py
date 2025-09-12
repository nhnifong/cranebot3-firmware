from vidgear.gears import CamGear
import cv2

# Define the stream and set low-latency options
options = {
    "STREAM_RESOLUTION": "1920x1080",
    "STREAM_FPS": 30,
    "fflags": "nobuffer",
    "flags": "low_delay"
}
stream = CamGear(
    source="tcp://192.168.1.157:8888", 
    logging=True,
)
stream.start()

try:
    while True:
        frame = stream.read()
        if frame is None:
            break
        
        # Process the frame here.
        cv2.imshow("Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    stream.stop()
    cv2.destroyAllWindows()
