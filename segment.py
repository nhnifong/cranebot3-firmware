from ultralytics import FastSAM
# from ultralytics import YOLO
import sys
import cv2

# Load a model
model = FastSAM("FastSAM-s.pt")
# model = YOLO("yolov8s-world.pt")

# Display model information (optional)
model.info()

# model.set_classes(["floor"])

cap = cv2.VideoCapture(sys.argv[1])

# Run inference
# results = model.track(source=sys.argv[1], imgsz=640, show=True, conf=0.8)
# results = model(source=sys.argv[1], show=True)

while True:
    ret, frame = cap.read()
    # if ret:
    #     dets = locate_markers(frame)
    #     for det in dets:
    #         cv2.drawFrameAxes(frame, mtx, distortion, np.array(det['r']), np.array(det['t']), 0.1, 6);
    #sframe = cv2.resize(frame, (2304, 1296),  interpolation = cv2.INTER_LINEAR)
    if ret:
        results = model.track(frame, conf=0.8, persist=True)

        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break