
# If you move from A to B without perfect info on anchor position, line length, and gantry position,
# there will be slack in one or more lines when you arrive at B.
# equalize the tension without moving the gantry and measure the motor angle changes required to equalize tension.
# Add those angle changes to the ones you made to make the A-B move to obtain the changes you would have had to make to
# move from one taut position to another taut position.

# what would the line lengths have been at start for us to have calculated the good move?
# calculate the reverse move using the new line lengths. Make that move. are the lines tight when you complete it?

# alternatively, try to optimize the whole collection of calibration parameters in one go.
# params include anchor poses, line reference lengths, zero angles.
# cost function terms
#	motor angle changes that would be taken to move the gantry from A to B, vs actual motor angle changes that were taken to move from A to B after tension equalization
#   difference between line lengths calculated from motor angle and line lengths calcualted from aruco gantry observation
# 	difference between pose calcuated from origin card observation and pose in param set
#   anchor pose z axis deviation from vertical