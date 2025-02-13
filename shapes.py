import numpy as np
import trimesh

def contour_combine(contours):
    """
    Combine the labelled contours of AI segmented images from multiple viewpoints into 3D shapes
    for each camera we have it's location, fov, and aspect ratio, which we can use to make a frustum the frame of reference of the room
    for each camera we have a list of class labels, and for each label, a list of contours representing the outline of the 
    region of the image that got classified with that lable.
    Create a polygon for each contour that projects it from the narrow to the wide end of the frustum. (a loft in onshape terminology)
    find the intersection of all the polygons of each class from all cameras with trimesh.boolean.intersection
    further filter these objects to throw out those that contain too little area.
    """
    pass