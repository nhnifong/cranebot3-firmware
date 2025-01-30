gravity=9.81

def sweet_spot(x):
    """
    A smooth function that describes the gripper sweet spot
    """
    a = 1.0  # Value at x=0
    b = -4.0 # Controls the rate of decay
    c = 0.1  # Target value at x=0.4
    d = 2 # shift x values so that the target values are at the desired x values

    return (a - c) * np.exp(b*(x**d)) + c

class grip_estimator:
    def __init__(self):
        self.body_position = BSpline()
        self.body_rotation = BSpline()
        self.payload_cog_position = BSpline()
        self.finger_angle = BSpline()

        # position of the grip spot relative to the gripper origin
        self.grip_spot_offset = np.array([0,0,-0.20])
        # time window size
        self.horizon_s = 10

    def grabbable(self):
        """
        returns a function that tells whether the payload would be in a grabble position relative to the gripper.
        the function gives a value between 0 and 1

        distance between payload and a point 20cm below the gripper origin
        passed through a nonlinear function
        """
        return lambda time: (
            sweet_spot(np.linalg.norm(
                self.body_position(time) + self.grip_spot_offset),
                self.payload_cog_position(time)))

    def plan_to_hold(self):
        """
        Returns a function that tells whether the robot plans to be holding an object at a give time

        This function is intended to be evaluated at future points and compared with other indicators of whether an object is held
        
        Only the condition under which we would grab and payload, and drop a payload matter.
        I suppose those are expectations of future events.

        so one way to do this would be to scan the time domain and check for those trigger conditions. The performance of that concerns me a little but it's probably
        something that can be dealt with later.
        """

        would_hold = 0
        for t in np.linspace(-self.horizon_s, self.horizon_s, 100):
            if self.grabbable(t):
                would_hold = True
            if self.over_bin(t):
                would_hold = false;

        return lambda time: (

        )

    def holding_payload(self, t):
        """
        Return whether measurements indicate that a payload is held at the time t

        Most reliable factors:
            there is a deviation between the commanded finger servo angle, and the actual angle read by an encoder.
            finger pressure is high. (remember that this is not a spline you can evaluate whenever you want)
            payload cog is near the grip spot
        """


    def desired_gripper_location(self, t):
        """
        Returns a timed point in the future where the gripper would like to be
        based on what it's holding at a given point in time. 


        is there any good reason to make this a function of time rather than always evaluate the present?
        if past data showed us holding something, but we aren't holding it anymore, any desire to move over a bin is now irrelevant
        there is no finger pressure or optical data about the future, but position splines may lead us to belive we will be holding something,
        even if we are not holding it now, and some time in the future after that, we would like to be at that object's bin. At best this may allow
        the minimizer to find position splines that swing the gripper over a thing, pick it up real fast, and already be moving towards the bin.
        Though honestly I doubt that'll work as well as it does in my imagination.
        """
        if self.holding_payload(t):
            # return a location over the bin appropriate for the payload we're holding.
        else:
            # return a location over the highest priority payload

        # when do you want to get there?
        # (distance to it / rough speed) seconds from now.
        # mind that I cant move up as as fast as I can move sideways or down
        # I don't care exactly where the gripper is between now and then, or where it is after that.