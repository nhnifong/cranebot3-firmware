from ursina import *
import model_constants
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation
from cv_common import compose_poses
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)

def to_ursina_rotation(rvec):
    euler = Rotation.from_rotvec(rvec).as_euler('xyz', degrees=True)
    return (-euler[0], -euler[2], euler[1])

class Ind(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            model='anchor',
            shader='lit_with_shadows_shader',
            **kwargs
        )
        self.x_thing = Entity(
            model='cube',
            scale=0.5,
            position=(1, 0, 0),
            color=color.red,
            parent=self,
        )
        self.y_thing = Entity(
            model='sphere',
            scale=0.5,
            position=(0, 1, 0),
            color=color.green,
            parent=self,
        )
        self.z_thing = Entity(
            model='diamond',
            scale=0.5,
            position=(0, 0, 1),
            color=color.blue,
            parent=self,
        )

app = Ursina()

EditorCamera()

axis_length = 1

# X-axis (red)
Entity(model='arrow', color=color.red, scale=(axis_length, 1, 1), rotation=(0, 0, -90))
Text('X', position=(axis_length + 1, 0, 0), color=color.red)

# Y-axis (green)
Entity(model='arrow', color=color.green, scale=(1, axis_length, 1), rotation=(0, 0, 0))
Text('Y', position=(0, axis_length + 1, 0), color=color.green)

# Z-axis (blue)
Entity(model='arrow', color=color.blue, scale=(1, 1, axis_length), rotation=(90, 0, 0))
Text('Z', position=(0, 0, axis_length + 1), color=color.blue)

# Example cube to show the axes in relation to an object
Ind(
    position=(2, 2, 2),
    color=color.white,
    )

def p(l):
    return np.array(l, dtype=float)

newpose = compose_poses([
    (p([0,0,pi/4]), p([0,0,0])),
    (p([0,pi/4,0]), p([0,0,0])),
])
print(newpose[0])

Ind(
    position=(-2, 2, 2),
    color=color.light_gray,
    rotation = to_ursina_rotation(newpose[0]),
    # rotation = to_ursina_rotation(model_constants.anchor_camera[0]),
    # rotation = to_ursina_rotation(np.array([0, 0, pi/4], dtype=float)),
    )

app.run()