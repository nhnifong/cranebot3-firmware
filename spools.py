import math
import asyncio
import time
import serial
import logging
import asyncio


DATA_LEN = 1000
PE_TERM = 1.5
# maximum acceleration in meters of line per second squared
MAX_ACCEL = 0.4
LOOP_DELAY_S = 0.03
# record line length every 10th iteration
REC_MOD = 10
TENSION_SMOOTHING_FACTOR = 0.2

# Constants used in tension equalization process
TENSION_SLACK_THRESH = 0.4 # kilograms. below this, line is assumed to be slack during tensioning
TENSION_TIGHT_THRESH = 0.7 # kilograms. above this, line is assumed to be too tight during tensioning.
MOTOR_SPEED_DURING_CALIBRATION = 1 # revolutions per second
MAX_LINE_CHANGE_DURING_CALIBRATION = 0.3 # meters
MKS42C_EXPECTED_ERR = 1.8 # degrees of expected angle error with no load per commanded rev/sec
MKS42C_TORQUE_FACTOR = 0.031 / 1.8 # kg-meters per degree of error. Factor for computing torque from the risidual angle error
MEASUREMENT_SPEED = -0.4 # speed at which to measure initial tension. slowest possible motor speed
MEASUREMENT_TICKS = 18 # number of 1/30 second ticks to measure to obtain a stable value.
DANGEROUS_TENSION = 2.5 # if this tension is exceeded, motion will stop.

def constrain(value, minimum, maximum):
    return max(minimum, min(value, maximum))

class SpoolController:
    def __init__(self, motor, empty_diameter, full_diameter, full_length, gear_ratio=1.0):
        """
        Create a controller for a spool of line.
        empty_diameter_mm is the diameter of the spool in millimeters when no line is on it.
        full_diameter is the diameter in mm of the bulk of wrapped line when full_length meters of line are wrapped.
        line_capacity_m is the length of line in meters that is attached to this spool.
            if all of it were reeled in, the object at the end would reach the limit switch, if there is one.
        gear_ratio refers to how many rotations the spool makes for one rotation of the motor shaft.
        """
        self.motor = motor
        self.empty_diameter = empty_diameter * 0.001
        self.full_diameter = full_diameter * 0.001
        self.full_length = full_length
        self.gear_ratio = gear_ratio
        self.motor_orientation = -1

        # since line accumulates on the spool in a spiral, the amount of wrapped line is an exponential function of the spool angle.
        self.diameter_diff = self.full_diameter - self.empty_diameter
        if self.diameter_diff > 0:
            self.k1_over_k2 = (self.empty_diameter * self.full_length) / self.diameter_diff
            self.k2 = (math.pi * self.gear_ratio * self.diameter_diff) / self.full_length
        else:
            self.k1_over_k2 = self.empty_diameter * self.full_length / 1e-9 # Avoid division by zero
            self.k2 = (math.pi * self.gear_ratio * 1e-9) / self.full_length

        # The motor shaft angle if the spool were empty, based on the last received reference length.
        self.zero_angle = 0
        
        # last commanded motor speed in revs/sec
        self.speed = 0
        self.lastLength = 0.5
        self.lastAngle = 0.0
        self.meters_per_rev = self.get_unspool_rate(self.lastAngle)
        # record of line length. tuples of (time, meters)
        self.record = []
        # plan of desired line lengths
        self.desiredLine = []
        self.lastIndex = 0
        self.runSpoolLoop = True
        self.rec_loop_counter = 0
        self.moveAllowed = True
        self.abort_equalize_tension = False
        self.live_tension_low_thresh = TENSION_SLACK_THRESH
        self.live_tension_high_thresh = TENSION_TIGHT_THRESH
        self.smoothed_tension = 0

        self.mks42c_expected_err = MKS42C_EXPECTED_ERR
        # use a specific value for this motor
        try:
            with open('mks42c_expected_err.cal', 'r') as f:
                self.mks42c_expected_err = float(f.read())
                logging.info(f'Used stored mks42c_expected_err value of {self.mks42c_expected_err }')
        except (FileNotFoundError, ValueError):
            print('Could not read saved mks42c_expected_err')
            pass

        # when this bool is set, spool tracking will pause.
        self.spoolPause = False

    def setReferenceLength(self, length):
        """
        Provide an external observation of the current unspooled line length
        """
        success = False
        attempts = 0
        while not success and attempts < 10:
            success, angle = self.motor.getShaftAngle()
            attempts += 1
        if success:
            # how many revs must the motor have turned since empty be to have this length
            spooled_length = self.full_length - length
            relative_angle = math.log(spooled_length / self.k1_over_k2 + 1) / self.k2
            angle *= self.motor_orientation
            self.zero_angle= angle - relative_angle
            logging.info(f'Reference length {length} m')
            logging.info(f'Set zero angle to be {self.zero_angle} revs. {relative_angle} revs from the current value of {angle}')

    def _get_spooled_length(self, motor_angle_revs):
        relative_angle = self.motor_orientation* motor_angle_revs - self.zero_angle # shaft angle relative to zero_angle
        if self.diameter_diff == 0:
            return relative_angle * self.gear_ratio * math.pi * self.empty_diameter
        else:
            return self.k1_over_k2 * (math.exp(self.k2 * relative_angle) - 1)

    def get_unspooled_length(self, motor_angle_revs):
        return self.full_length - self._get_spooled_length(motor_angle_revs)

    def get_unspool_rate(self, motor_angle_revs):
        relative_angle = self.motor_orientation * motor_angle_revs - self.zero_angle

        if self.diameter_diff == 0:
            return math.pi * self.empty_diameter * self.gear_ratio
        else:
            rate_spool_rev_per_spool_rev = math.pi * (self.empty_diameter + self.diameter_diff * (self._get_spooled_length(motor_angle_revs) / self.full_length))
            return rate_spool_rev_per_spool_rev * self.gear_ratio

    def commandSpeed(self, speed):
        """command a specific speed from the motor."""
        self.speed = speed
        self.motor.runConstantSpeed(self.speed)


    def setPlan(self, plan):
        """
        set plan to an array of tuples of time and length
        """
        self.desiredLine = plan
        self.lastIndex = 0

    def jogRelativeLen(self, rel):
        new_l = self.lastLength + rel
        self.setPlan([
            (time.time(), self.lastLength),
            (time.time()+2, new_l),
        ])

    def popMeasurements(self):
        copy_record = self.record
        self.record = []
        return copy_record

    def currentLineLength(self):
        """
        return the current time and current unspooled line in meters
        Also store the length in an array to be popped later.
        """
        success, angle = self.motor.getShaftAngle()
        if not success:
            logging.warning("Could not read shaft angle from motor")
            return (time.time(), self.lastLength, self.smoothed_tension)

        if abs(angle - self.lastAngle) > 1:
            logging.warning(f'motor moved more than 1 rev since last read, lastAngle={self.lastAngle} angle={angle} diff={angle - self.lastAngle}')
        self.lastAngle = angle

        self.lastLength = self.get_unspooled_length(angle)
        self.meters_per_rev = self.get_unspool_rate(angle)
        # logging.debug(f'current unspooled line = {self.lastLength} m. rate = {self.meters_per_rev} m/r')

        # self.moveAllowed = True
        # if self.lastLength < 0 or self.lastLength > self.full_length:
        #     logging.error(f"Bad length calculation! length={self.lastLength}, shaftAngle={angle}. Movement disallowed until new reference length received.")
        #     self.moveAllowed = False

        self.smoothed_tension = self.currentTension() * TENSION_SMOOTHING_FACTOR + self.smoothed_tension * (1-TENSION_SMOOTHING_FACTOR)
        # if self.smoothed_tension > DANGEROUS_TENSION:
        #     logging.warning(f"Tension of {self.smoothed_tension} is too high!")
        #     # try to loosen up right away to avoid breaking something
        #     self.commandSpeed(1)
        #     time.sleep(1)
        #     self.commandSpeed(0)
        #     self.moveAllowed = False

        # accumulate these so you can send them to the websocket
        row = (time.time(), self.lastLength, self.smoothed_tension)
        if self.rec_loop_counter == REC_MOD:
            self.record.append(row)
            self.rec_loop_counter = 0
        self.rec_loop_counter += 1
        return row

    def currentTension(self):
        """return the line tension in kilograms of force.
        Only designed for the MKS42C
        Measurements are basically invalid below speeds of 0.4
        """
        _, angleError = self.motor.getShaftError()
        baseline = self.mks42c_expected_err * self.speed
        load_err = baseline - angleError # the residual position error attributable to load on the spool, not commanded motor speed
        # greater load on the line subtracts from angleError regardless of the commanded motor direction.
        ratio = load_err / baseline
        special = load_err
        if self.speed < 0: special /= abs(self.speed-1)
        print(f'currentTension() baseline={baseline:.3f} load_err={load_err:.3f} ratio={ratio:.3f} special={special:.3f}')
        torque = MKS42C_TORQUE_FACTOR * load_err
        return torque / self.meters_per_rev

    def measure_t(self):
        try:
            while True:
                self.commandSpeed(math.sin(time.time()/4)*2)
                t, l, tension = self.currentLineLength()
                print(f'tension = {tension:.3f}')
                time.sleep(1/30)
        except KeyboardInterrupt:
            self.commandSpeed(0)

    def fastStop(self):
        # fast stop is permanent.
        # it causes the trackingloop task to stop,
        # causing the websocket connection to close
        self.motor.stop()
        self.runSpoolLoop = False

    def pauseTrackingLoop(self):
        self.spoolPause = True

    def resumeTrackingLoop(self):
        self.spoolPause = False

    def trackingLoop(self):
        """
        Constantly try to match the position and speed given in an array

        """
        while self.runSpoolLoop:
            if self.spoolPause:
                time.sleep(0.2)
                continue
            try:
                t, currentLen, tension = self.currentLineLength()

                # Find the earliest entry in desiredLine that is still in the future.
                while self.lastIndex < len(self.desiredLine) and self.desiredLine[self.lastIndex][0] <= t:
                    self.lastIndex += 1

                # slow stop when there is no data to track
                if self.lastIndex >= len(self.desiredLine):
                    if abs(self.speed) > 0:
                        newspeed = self.speed * 0.9
                        if abs(newspeed) < 0.2:
                            newspeed = 0
                        logging.debug(f'Slow stopping {newspeed}')
                        self.commandSpeed(newspeed)
                    time.sleep(LOOP_DELAY_S)
                    continue

                targetLen = self.desiredLine[self.lastIndex][1]
                position_err = targetLen - currentLen

                # What would the speed be between the two datapoints tha straddle the present?
                # Positive values mean line is lengthening
                # result is in meters of line per second
                # if there is only one desired length, we can't find two that straddle the present and have to use current length and time
                if self.lastIndex == 0:
                    last_time = t
                    last_len = currentLen
                else:
                    last_time = self.desiredLine[self.lastIndex-1][0]
                    last_len = self.desiredLine[self.lastIndex-1][1]
                targetSpeed = ((self.desiredLine[self.lastIndex][1] - last_len)
                    / (self.desiredLine[self.lastIndex][0] - last_time))
                # change in line length in meters per second
                currentSpeed = self.speed * self.meters_per_rev
                speed_err = targetSpeed - currentSpeed
                # If our positional error was zero, we could go exactly that speed.
                # if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
                # (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
                # ideally we want to catch up in one step, but we have max acceleration constraints.
                aimSpeed = targetSpeed + position_err * PE_TERM; # meters of line per second

                # limit the acceleration of the line
                wouldAccel = (aimSpeed - currentSpeed) / LOOP_DELAY_S
                if wouldAccel > MAX_ACCEL:
                    aimSpeed = MAX_ACCEL * LOOP_DELAY_S + currentSpeed
                elif wouldAccel < -MAX_ACCEL:
                    aimSpeed = -MAX_ACCEL * LOOP_DELAY_S + currentSpeed

                maxspeed = self.motor.getMaxSpeed()

                if self.moveAllowed:
                    self.commandSpeed(constrain(aimSpeed / self.meters_per_rev, -maxspeed, maxspeed))
                else:
                    logging.warning(f"would move at speed={self.speed} but length is invalid. calibrate length first.")

                time.sleep(LOOP_DELAY_S)
            except serial.serialutil.SerialTimeoutException:
                logging.error('Lost serial contact with motor')
                break

    async def measureNoLoad(self):
        """
        Obtain an estimate of the expected angle error with no load per commanded rev/sec specific to this motor.
        """
        self.pauseTrackingLoop()
        data = []
        for speed in [-0.4, 1, 0.4, -1]:
            self.commandSpeed(speed)
            for i in range(18):
                valid, err = self.motor.getShaftError()
                if valid:
                    data.append(err / speed)
                await asyncio.sleep(1/30)
        self.commandSpeed(0)
        self.mks42c_expected_err = sum(data)/len(data)
        print(f'calibrated mks42c_expected_err = {self.mks42c_expected_err} deg')
        with open('mks42c_expected_err.cal', 'w') as f:
            f.write(f'{self.mks42c_expected_err}\n')
        self.resumeTrackingLoop()

    async def equalizeSpoolTension(self,
        controllerApprovalEvent,
        sendUpdatesFunc,
        maxLineChange=MAX_LINE_CHANGE_DURING_CALIBRATION,
        allowOutSpooling=True):
        """Without tracking any particular length, reel the spool until the line tension is within a predefined range

        Pause spool tracking loop
        Measure the starting shaft angle
        measure shaft angle error. Very low errors indicate slackness, higher errors indicate tightness.
        While (angle error is not in the middle of it's range) and (calibration not aborted) and (line change is less than safe threshold):
            run the motor at a constant very low speed (positive for tight lines, negative for slack lines)
            Measure the shaft angle and compute the change in the amount of outspooled line since start
        report the value to the controller and remain paused.
        When the controller approves,
        Change the reference length by that amount.
        Unpause the tracking loop.

        Any lines that started slack will begin to reel in
        Any line that didn't start slack will begin to reel out.
        reeling (out or in) should immediately stop if
          1. the slackness threshold is crossed (we can see this locally)
        reeling out should also stop if
          2. Any other line that started slack has stopped (controller will inform us)
        """
        self.pauseTrackingLoop()
        # make sure the tracking loop is definitely paused. it's on another thread. 
        await asyncio.sleep(LOOP_DELAY_S * 2)
        self.commandSpeed(0)
        self.abort_equalize_tension = False
        t, curLength, tension = self.currentLineLength()
        startLength = curLength
        line_delta = 0

        # reset thresholds to default
        self.live_tension_low_thresh = TENSION_SLACK_THRESH
        self.live_tension_high_thresh = TENSION_TIGHT_THRESH

        logging.info("Measuring tension in motion")
        # measuring tension at rest is not possible. measure in motion
        self.commandSpeed(MEASUREMENT_SPEED)
        for i in range(MEASUREMENT_TICKS):
            t, curLength, tension = self.currentLineLength()
            logging.debug(f'getting stabilized tension reading {self.smoothed_tension}')
            await asyncio.sleep(1/30)
        # undo motion that occurred during reading
        self.commandSpeed(-MEASUREMENT_SPEED)
        await asyncio.sleep(MEASUREMENT_TICKS/30)
        self.commandSpeed(0)

        # decide initial speed. motor direction will not change during the loop
        if self.smoothed_tension < TENSION_SLACK_THRESH:
            started_slack = True
            self.commandSpeed(-MOTOR_SPEED_DURING_CALIBRATION)
        else:
            started_slack = False
            self.commandSpeed(MOTOR_SPEED_DURING_CALIBRATION)
        is_slack = started_slack

        logging.info(f'Started slack={started_slack} with a tension of {self.smoothed_tension} kg at a measurement speed of {MEASUREMENT_SPEED} motor revs/s')
 
        try:
            # wait for stop condition
            while ((started_slack == is_slack)
                   and not self.abort_equalize_tension
                   and abs(line_delta) < maxLineChange
                   and (started_slack or allowOutSpooling)
                   and self.smoothed_tension < DANGEROUS_TENSION):
                await asyncio.sleep(1/30)
                # self.currentLineLength() causes length and tension to be calculated and recorded in a list that
                # is periodically flushed to the websocket by a task that is always running while the ws is connected
                t, curLength, tension = self.currentLineLength()
                logging.debug(f'curLength={curLength} tension={tension}')
                line_delta = curLength - startLength
                is_slack = self.smoothed_tension < self.live_tension_low_thresh
        except e:
            self.commandSpeed(0)
            raise e

        # todo, verify by experiment
        # if you started tight but became slack, reel back in a small amount

        # inform controller that we hit a trigger and stopped.
        sendUpdatesFunc({
            'tension_seek_stopped': {
                'line_delta': line_delta,
                'started_slack': started_slack,
                'is_slack': is_slack,
            }
        })

        logging.info(f"Stopped equalization with tension={self.smoothed_tension} and a line delta of {line_delta}")
        self.commandSpeed(0)
        try:
            logging.info('Waiting for tension_eq result approval from controller')
            await asyncio.wait_for(controllerApprovalEvent.wait(), timeout=30)
            # the caller will have set this flag while we were waiting to indicate approval.
            if not self.abort_equalize_tension:
                self.setReferenceLength(startLength)
        except TimeoutError:
            pass
        self.desiredLine = []
        self.resumeTrackingLoop()
