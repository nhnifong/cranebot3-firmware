# draft of gripping procedure

gantry_lower_z = 1.2
small_up = np.array([0,0,0.15])

async def grab_object(self, position):
	# position represents an estimate of where the object is.
	# position gripper 15 cm over object
	# position gantry at the bottom of it's vertical safe box
	gripper_pos = position + small_up
	gantry_pos = position.copy()
	gantry_pos[2] = gantry_lower_z

	# start gantry movement.
	self.gantry_goal_pos = gantry_pos
	goalseek_task = asyncio.create_task(self.seek_gantry_goal())

	# start winch movement
    plan = np.array([[soon, target_winch_len]])
    message = {'length_plan' : plan.tolist()}
    winch_and_wait_task = asyncio.create_task(client.send_commands(message))

    # wait for movement to finish
    await asyncio.gather([goalseek_task, winch_and_wait_task])

    # wait for swinging to stop

    # measure where object is in gripper view
    object_view_coords
    # measure distance to center
    center_error np.linalg.norm(object_view_coords - np.array([0.5, 0.5]))
    # using gripper rotation from IMU, get vector to move towards center.

    # move towards center and wait for motion to stop.

    # drop winch until laser rangefinder shows 5 cm.

    # start grab and wait for result of first attempt

    # got something?
	    # winch up and see if it falls out
	# else
		# hop in random direction and try again