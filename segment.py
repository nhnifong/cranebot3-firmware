from ultralytics import FastSAM
import sys
import cv2
import numpy as np
from trimesh.creation import extrude_polygon, triangulate_polygon
from trimesh import Trimesh
from shapely.geometry import Polygon
import math
import torch



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
    model = FastSAM("FastSAM-s.pt")

    # for results in model(sys.argv[1], stream=True, texts="a peice of clutter", conf=0.8):
    for results in model.track(sys.argv[1], stream=True, conf=0.75, save=False):
        # make_shapes(results[0].masks.xyn)
        if results:
            areas = results.boxes.xywhn[:, 2] * results.boxes.xywhn[:, 3]
            if results.boxes.id is None:
                continue
            box_sizes = torch.stack((results.boxes.id, areas), dim=1)
            filtered_masks = results.masks.xyn[box_sizes[:, 1] <= 0.002]
            # results.orig_img
            shapes = make_shapes(filtered_masks)
            # print(results.masks)

        annotated_frame = results.plot()

        # int_points = results[0].masks.xy[0].astype(np.int32)
        # cv2.polylines(frame, [int_points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# cont = np.load('cont.npz')['cont']
# make_shapes([cont])

stream()


# On every frame received from a camera, perform inference on that frame with the model, (ignore persist=True for now, there is probably some way to context switch)
# for every detected object, check that it is within the right size range.