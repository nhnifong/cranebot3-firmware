from ultralytics import FastSAM
import sys
import cv2
import threading
import numpy as np
from trimesh import Trimesh
from trimesh.creation import extrude_polygon, triangulate_polygon
from trimesh.transformations import compose_matrix, translation_matrix
from trimesh.boolean import intersection
from trimesh.collision import CollisionManager
from shapely.geometry import Polygon
import math
import torch
from scipy.cluster.hierarchy import DisjointSet

def rodrigues_matrix(rvec):
    """
    Creates a 4x4 rotation matrix from a Rodrigues vector (rvec)
    as returned by cv2.solvePnP.
    """
    rotation_matrix_3x3, _ = cv2.Rodrigues(rvec)
    # Convert 3x3 to 4x4 matrix
    rotation_matrix_4x4 = np.eye(4)  # Identity matrix
    rotation_matrix_4x4[:3, :3] = rotation_matrix_3x3
    return rotation_matrix_4x4


class ShapeTracker:
    def __init__(self):
        self.model = FastSAM("FastSAM-s.pt")
        self.collision_manager = CollisionManager()
        self.size_thresh = 0.003
        self.last_shapes_by_camera = [{}, {}, {}, {}]
        self.camera_transforms = []


        # after extrusion the shape's narrow end is at z=0 and it extends up along the z axis.
        # The narrow end is still in the normalized units the contours were in.
        # since the contours were normalized, the width of the frame is 1 unit, so the truncated prism's
        # narrow end should be placed 1 meter from the camera in world space.
        # TODO find out if x and y were normalized independently, or only based on the larger dimension.
        # if it was independent, then restore the original aspect ratio
        # use the camera matrix to undo any spherical distortion
        self.standard_transform = translation_matrix([0,0,1]) # move up 1 meter
        

    def setCameraPoses(self, poses):
        self.camera_transforms = [
            compose_matrix(
                translation_matrix(tvec),
                rodrigues_matrix(rvec),
            )
            for rvec,tvec in poses]

    def processFrame(self, anchor_num, frame):
        results = self.model(frame, conf=0.75, imgsz=(1138,640), save=False)
        filtered_masks = []
        if results:
            # pick out only objects that are small
            r = results[0]
            area_mask = r.boxes.xywhn[:, 2] * r.boxes.xywhn[:, 3] < self.size_thresh
            filtered_masks = [r.masks.xyn[i] for i in range(len(r.masks.xyn)) if area_mask[i]]
            print(f'cam {anchor_num} showing {len(filtered_masks)} objects with an area smaller than {self.size_thresh}')
        self.last_shapes_by_camera[anchor_num] = self.make_shapes(anchor_num, filtered_masks)

        # trigger the shape merge once per cycle, presumably we receive frames from the other three cameras
        if anchor_num == 0:
            mgd = {}
            for shapes in self.last_shapes_by_camera:
                mgd.update(shapes)
            r = self.merge_shapes(mgd)
            print(r)

    def make_shapes(self, anchor_num, contours):
        """Project contours from a particular camera into shapes in world space
        The shapes are like a truncated prism with the outline of the contour and the tapering
        ratio of a frustum
        contours = [[x, y],...]
        """
        height = 6
        fov = 1.15192 # radians
        shapes = {}
        for object_id, contour in enumerate(contours):
            shape = extrude_polygon(Polygon(contour), height=height)
            top_scale = math.sin(fov)*height
            for i in range(len(shape.vertices)):
                if shape.vertices[i][2] == 0:
                    continue
                shape.vertices[i][0] *= top_scale
                shape.vertices[i][1] *= top_scale
            assert(shape.is_watertight)
            shape.apply_transform(self.standard_transform)
            shape.apply_transform(self.camera_transforms[anchor_num])
            shapes[f'{anchor_num}-{object_id}'] = shape
        return shapes

    def merge_shapes(self, shapes):
        """
        Take a dictionary of shapes with ids, and find the intersections.
        key structure must be f'{anchor_num}-{object_id}'

        We know we can make ids stable between subsequet pictures from a single camera
        Put all the shapes in the trimesh CollisionManager named by their anchor_num-id,
        request the full list of collisions
        initially throw out collisions that do not occur close to the floor.
        Collisions may have multiple contacts, but just look at the z of contacts[0].point
        Find the disjoint sets of node pairs. (scipy.cluster.hierarchy.DisjointSet)
        initially every node is it's own set. for ever collision pair, merge the sets that those two nodes belong to.
        Once complete, for every subset in the disjoint set containing 2 or more nodes, take the boolean intersection of all
        the original shapes those nodes represent. Keep any objects that have volume
        tag these resulting objects with the original ids from any views they were observed in,
        and any text prompts that were used to select them
        """
        results = []
        ds = DisjointSet()
        for name, mesh in shapes.items():
            self.collision_manager.add_object(name, mesh)
            ds.add(name)
        aru, names, data = self.collision_manager.in_collision_internal(return_names=True, return_data=True)
        if aru:
            for name_pair, collision in zip(names, data):
                # disregard collisions that occured more than 1 meter from the floor
                if collision.point[2] > 1:
                    continue
                # disregard collisions of two shapes from the same camera
                cam_numbers = [name.split('-')[0] for name in name_pair]
                if cam_numbers[0] == cam_numbers[1]:
                    continue
                ds.merge(*name_pair)
            for subset in ds.subsets():
                if len(subset) > 1:
                    final_shape = intersection([shapes[name] for name in subset], engine='manifold')
                    results.append({
                        'mesh': final_shape,
                        'original_ids': list(subset)
                    })
        return results


def stream():
    model = FastSAM("FastSAM-s.pt")

    # for results in model(sys.argv[1], stream=True, texts="a peice of clutter", conf=0.8):
    for results in model.track(sys.argv[1], stream=True, conf=0.75, save=False, texts="a silly kid"):
        # make_shapes(results[0].masks.xyn)
        # if results:
        #     areas = results.boxes.xywhn[:, 2] * results.boxes.xywhn[:, 3]
        #     if results.boxes.id is None:
        #         continue
        #     box_sizes = torch.stack((results.boxes.id, areas), dim=1)
        #     filtered_masks = results.masks.xyn[box_sizes[:, 1] <= 0.002]
        #     # results.orig_img
        #     shapes = make_shapes(filtered_masks)
        #     # print(results.masks)

        annotated_frame = results.plot()

        # int_points = results[0].masks.xy[0].astype(np.int32)
        # cv2.polylines(frame, [int_points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def frame_generator(num):
    video_uri = f'tcp://192.168.1.{num}:8888'
    print(f'Connecting to {video_uri}')
    cap = cv2.VideoCapture(video_uri)
    i = 0
    while True:
        ret, frame = cap.read()
        if ret and i==0:
            yield frame
        i = (i+1)%30

def receive_video(num, st):
    for frame in frame_generator(num):
        try:
            st.processFrame(num-151, frame)
        except KeyboardInterrupt:
            return

# cont = np.load('cont.npz')['cont']
# make_shapes([cont])

# stream()


# mod = FastSAM("FastSAM-s.pt")
# st = ShapeTracker()

# vid_1 = threading.Thread(target=receive_video, args=(151, st))
# vid_2 = threading.Thread(target=receive_video, args=(153, mod))
# vid_1.start()
# vid_2.start()

# receive_video(153, st)