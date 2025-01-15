import numpy as np
from scipy.interpolate import BSpline

control_points = np.array([
	[2,5,3],
	[1,2,0],
	[4,1,8],
	[7,2,7],
])

# I know you can shift the domain of a spline by moving all the knots, but can you leave the knots at 0-1 and shift it by moving all the control points?

spline_degree = 3
base_interval = (0, 1)
clamped_knots = np.array([0,0,0,0,1,1,1,1])
spl = BSpline(clamped_knots, control_points, spline_degree)

# make a new spline with the exact same knots such that when evaluated over (offset, 1+offset) it produces the same curve as the
# first one evaluated over (0, 1)
# the first and last control points of the new spline can be created by sampling the first spline at offsets, but how about the other two?



## AI code follows

def translate_control_points(control_points, knots, degree, offset):
    """
    Computes new control points for a B-spline translation.

    Args:
        control_points: Original control points.
        knots: Knot vector.
        degree: Spline degree.
        offset: Translation offset.

    Returns:
        New control points.
    """

    num_control_points = len(control_points)
    num_eval_points = num_control_points*10 # Higher than control points for better accuracy

    eval_points = np.linspace(0, 1, num_eval_points)
    eval_matrix = BSpline.design_matrix(eval_points, knots, degree)
    
    shifted_eval_points = eval_points - offset
    shifted_eval_matrix = BSpline.design_matrix(shifted_eval_points, knots, degree)

    new_control_points = np.linalg.lstsq(eval_matrix, shifted_eval_matrix @ control_points, rcond=None)[0] # Use least squares because eval_matrix is not square

    return new_control_points

offset = 0.5
new_control_points = translate_control_points(control_points, clamped_knots, spline_degree, offset)
spl_translated = BSpline(clamped_knots, new_control_points, spline_degree)

import matplotlib.pyplot as plt
x = np.linspace(0, 1, 100)
x_translated = np.linspace(offset, 1 + offset, 100)

plt.plot(x, spl(x), label="Original Spline")
plt.plot(x_translated, spl(x), label="Original Spline shifted")
plt.plot(x, spl_translated(x), label="Translated Spline")
plt.legend()
plt.show()
