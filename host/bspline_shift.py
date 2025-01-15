import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(original_path[:, 0], original_path[:, 1], original_path[:, 2], label="Original Spline (0-1)")
# ax.plot(extended_path[:, 0], extended_path[:, 1], extended_path[:, 2], label="Extended Spline (0-2)")
ax.scatter(control_points[:,0], control_points[:,1], control_points[:,2], c='red', label='Control Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('B-Spline')
ax.legend()
plt.save("foo.png")