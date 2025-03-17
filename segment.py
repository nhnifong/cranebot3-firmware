from ultralytics import FastSAM
import sys
import cv2
import threading
import numpy as np
import trimesh
from itertools import combinations
from shapely.geometry import Polygon
import math
import torch
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial import ConvexHull

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

def simplify_contour_scipy_convexhull(contour):
    """Simplifies a contour using scipy.spatial.ConvexHull.
    Expect about a 10:1 reduction and almost total elimination of boolean op errors"""
    hull = ConvexHull(contour.squeeze()) #remove extra dimension.
    simplified_contour = contour[hull.vertices]
    return simplified_contour


class ShapeTracker:
    def __init__(self):
        self.model = FastSAM("FastSAM-s.pt")

        self.size_thresh = 0.05
        self.confidence_thresh = 0.81
        self.max_height = 0.2 # height limit in meters of detected objects
        self.min_views = 2 # minimum number of views an object must be seen from. 2 or 3
        self.preferred_delay = 0.6 # sec
        self.last_shapes_by_camera = [{}, {}, {}, {}]
        self.camera_transforms = [None, None, None, None]


        # after extrusion the shape's narrow end is at z=0 and it extends up along the z axis.
        # The narrow end is still in the normalized units the contours were in.
        # since the contours were normalized, the width of the frame is 1 unit, so the truncated prism's
        # narrow end should be placed 1 meter from the camera in world space.
        # TODO find out if x and y were normalized independently, or only based on the larger dimension.
        # if it was independent, then restore the original aspect ratio
        # use the camera matrix to undo any spherical distortion
        self.standard_transform = trimesh.transformations.translation_matrix([0,0,1]) # move up 1 meter
        

    def setCameraPoses(self, anchor_num, pose):
        print(f'shape tracker setting pose for camera {anchor_num} to {pose}')
        self.camera_transforms[anchor_num] = np.dot(
            trimesh.transformations.translation_matrix(pose[1]),
            rodrigues_matrix(pose[0]))

    def processFrame(self, anchor_num, frame):
        results = self.model(frame, conf=self.confidence_thresh, save=False, imgsz=2048,) 
        yratio = frame.shape[0] / frame.shape[1]
        filtered_masks = []
        if results:
            # pick out only objects that are small
            r = results[0]
            if r.masks is not None:
                area_mask = r.boxes.xywhn[:, 2] * r.boxes.xywhn[:, 3] < self.size_thresh
                filtered_masks = []
                for i in range(len(r.masks.xyn)):
                    if area_mask[i]:
                        # fix the aspect ratio of the mask
                        r.masks.xyn[i] += np.array([-0.5, -0.5])
                        r.masks.xyn[i][:,1] *= yratio
                        filtered_masks.append(r.masks.xyn[i])
        self.last_shapes_by_camera[anchor_num] = self.make_shapes(anchor_num, filtered_masks)

    def make_shapes(self, anchor_num, contours):
        """Project contours from a particular camera into shapes in world space
        The shapes are like a truncated prism with the outline of the contour and the tapering
        ratio of a frustum
        contours = [[x, y],...]
        """
        height = 6
        fov = 1.15192 # horizontal field of view in radians
        shapes = {}
        collision_manager = trimesh.collision.CollisionManager()
        for object_id, contour in enumerate(contours):
            # simplified_contour = cv2.approxPolyDP(contour, 2.0, True)
            simplified_contour = simplify_contour_scipy_convexhull(contour)
            shape = trimesh.creation.extrude_polygon(Polygon(simplified_contour), height=height)
            top_scale = math.sin(fov)*height
            for i in range(len(shape.vertices)):
                if shape.vertices[i][2] == 0:
                    continue
                shape.vertices[i][0] *= top_scale
                shape.vertices[i][1] *= top_scale
            try:
                assert(shape.is_watertight)
                shape.apply_transform(self.standard_transform)
                if self.camera_transforms[anchor_num] is not None:
                    shape.apply_transform(self.camera_transforms[anchor_num])
                key = f'{anchor_num}-{object_id}'
                shapes[key] = shape
                collision_manager.add_object(key, shape)
            except AssertionError:
                print('discarded a shape from a contour because it was not watertight')

        # no overlapping shapes are allowed. they trash the downstream boolean operations
        aru, names = collision_manager.in_collision_internal(return_names=True)
        for name_pair in names:
            # delete one
            try:
                del shapes[name_pair[1]]
                collision_manager.remove_object(name_pair[1])
            except KeyError:
                pass
        # aru = collision_manager.in_collision_internal()
        # assert(not aru)

        return shapes

    def merge_shapes(self):
        """
        for each camera, concatenate the shapes together. trimesh.util.concatenate([mesh1, mesh2])
        intersect each of these four shapes with a flat rectangle that rises about 20cm from the floor.
        you have one flat multipart shape for each camera containing things near the floor.
        create an intersection for every pair of cameras.
        an intersection for every triplet of cameras
        and an intersection for all four cameras.
        split every resulting intersection object by disconnected parts mesh.graph.connected_components()

        the problem with this algorithm seems to be that there are shapes from the same camera that overlap
        so when [floorbox, concat] are intersected, you get a non-watertight volume.
        """
        floorbox = trimesh.creation.box([10, 10, self.max_height])
        floorbox.apply_translation([0, 0, self.max_height / 2])
        floorshapes = []
        for i,shapes in enumerate(self.last_shapes_by_camera):
            if len(shapes) > 0:
                concat  = trimesh.util.concatenate(shapes.values())
                # concat = trimesh.boolean.union(shapes.values(), engine='manifold')
                try:
                    x = trimesh.boolean.intersection([floorbox, concat], engine='manifold')
                    floorshapes.append(x)
                except ValueError:
                    pass
            # moving the floorbox a tiny bit helps avoid coplanar faces
            floorbox.apply_translation([0.0001, 0.0001, 0.0001])

        if len(floorshapes) <= 1:
            return {}

        solids = {} # key: number of cameras seen in, value, list of meshes
        for set_size in range(self.min_views, len(floorshapes)+1):
            solids[set_size] = []
            for indices in combinations(set(range(len(floorshapes))), set_size):
                try:
                    flats = trimesh.boolean.intersection([floorshapes[i] for i in indices], engine='manifold')
                    solids[set_size].extend(trimesh.graph.split(flats))
                except ValueError as e:
                    # for i,fs in enumerate(floorshapes):
                    #     fs.export(f'debug_solids_{i}.stl')
                    print(f'could not intersect objects from cameras {indices} {e}')
            # print(f'{len(solids[set_size])} objects are visible from {set_size} cameras')
        return solids


        # intersection is kind of slow, hence the funny method here instead of the elegant way
        # if len(floorshapes) > 2:
        #     seen_thrice = [trimesh.boolean.intersection([seen_twice[0], floorshapes[2]], engine='manifold')]
        #     if len(floorshapes) > 3:
        #         # we have (0, 1, 2), we also need (0, 1, 3), (0, 2, 3), (1, 2, 3)
        #         seen_thrice.extend([trimesh.boolean.intersection([seen_twice[sti], floorshapes[fsi]], engine='manifold')
        #             for sti, fsi in [(0, 3), (1, 3), (3, 3)]])
        #         seen_force = [trimesh.boolean.intersection([seen_twice[0], seen_twice[-1]], engine='manifold')]



    def merge_shapes_with_disjoint_se(self):
        """
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


        the problem with this algorithm seems to be that it doesn't take too many collisions for all the shapes to end up in the same disjoint set
        and when they intersect, there's nothing in common between all of them.
        you might have the set {'3-14', '0-0', '3-7', '1-7'} for example becase 0-0 shot through two shapes from cam 3 and one shape from cam 1.
        we are never interested in an intersection between two shapes from the same camera.

        Another problem seems to be that the collision engine doesn't report collisions every time, even if the shapes are never moving.
        But the collision manager is fast and cheap, so maybe running it more than once will help  

        We are only interested in taking intersections of the outlines of objects if they really are the same object from different angles.
        The only reliably heuristic we have that indicates they are the same object is the location on the floor.
        find the point where the ray in the center of the object's bounding box intersects the floor.
        For all shape pairs, regardless of whether they collide, if their floor points are close, take their intersection.
        """

        # trigger the shape merge once per cycle, presumably we receive frames from each camera
        mgd = {}
        for shapes in self.last_shapes_by_camera:
            mgd.update(shapes)
        # mgd is a dictionary of shapes with ids
        # key structure is f'{anchor_num}-{object_id}'

        solids = []
        # ds = DisjointSet()
        collision_manager = trimesh.collision.CollisionManager()
        for name, mesh in mgd.items():
            collision_manager.add_object(name, mesh)
            # ds.add(name)
        aru, names, data = collision_manager.in_collision_internal(return_names=True, return_data=True)
        if aru:
            print(f'examining {len(names)} collision candidates')
            for name_pair, collision in zip(names, data):
                # disregard collisions that occured more than 1 meter from the floor
                if abs(collision.point[2]) > 1:
                    print(f'Collision discarded for being too far away from the floor ({collision.point[2]})')
                    continue
                # disregard collisions of two shapes from the same camera
                cam_numbers = [name.split('-')[0] for name in name_pair]
                if cam_numbers[0] == cam_numbers[1]:
                    print(f'Collision {name_pair} discarded for not involving two unique camera views')
                    continue
                print(f'collision valid {name_pair}')
                # solids.append(intersection([mgd[name] for name in name_pair], engine='manifold'))
                ds.merge(*name_pair)
            print(f'Disjoint subsets {ds.subsets()}')
            for subset in ds.subsets():
                if len(subset) > 1:
                    print('Intersecting shape')
                    final_shape = trimesh.boolean.intersection([mgd[name] for name in subset], engine='manifold')
                    # I may have some reason in the future to know the ids
                    # solids.append({
                    #     'mesh': final_shape,
                    #     'original_ids': list(subset)
                    # })
                    solids.append(final_shape)
        return solids


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