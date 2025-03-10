from ultralytics import FastSAM
# from ultralytics import YOLO
import sys
import cv2
import numpy as np
from trimesh.creation import extrude_polygon, triangulate_polygon
from trimesh import Trimesh
from trimesh.transformations import clip_matrix, inverse_matrix, projection_matrix
from shapely.geometry import Polygon
import math



# Run inference
# results = model.track(source=sys.argv[1], imgsz=640, show=True, conf=0.8)
# results = model(source=sys.argv[1], show=True)

def make_shapes(contours):
    height = 6
    fov = 1.15192 # radians
    for contour in contours:
        shape = extrude_polygon(Polygon(contour), height=height)
        top_scale = math.sin(fov)*height
        for i in range(len(shape.vertices)):
            if shape.vertices[i][2] == 0:
                continue
            shape.vertices[i][0] *= top_scale
            shape.vertices[i][1] *= top_scale
        assert(shape.is_watertight)
    # shape.show()

def stream():
    # Load a model
    model = FastSAM("FastSAM-s.pt")
    # model = YOLO("yolov8s-world.pt")
    model.info()
    cap = cv2.VideoCapture(sys.argv[1])
    while True:
        ret, frame = cap.read()
        # if ret:
        #     dets = locate_markers(frame)
        #     for det in dets:
        #         cv2.drawFrameAxes(frame, mtx, distortion, np.array(det['r']), np.array(det['t']), 0.1, 6);
        #sframe = cv2.resize(frame, (2304, 1296),  interpolation = cv2.INTER_LINEAR)
        if ret:
            results = model.track(frame, conf=0.8, persist=True)
            # print(results[0].masks.xyn[0]) # it's already a contour!

            # make_shapes(results[0].masks.xyn)

            annotated_frame = results[0].plot()
            # int_points = results[0].masks.xy[0].astype(np.int32)
            # cv2.polylines(frame, [int_points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Display the annotated frame
            cv2.imshow("Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cont = np.load('cont.npz')['cont']
make_shapes([cont])