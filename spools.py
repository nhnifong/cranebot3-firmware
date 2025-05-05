import math
import asyncio
import time
import serial
import logging
import asyncio

# values that can be overridden by the controller
default_conf = {
    # number of records of length and tension to keep
    'DATA_LEN': 1000,
    # factor controlling how much positon error matters in the tracking loop.
    'PE_TERM': 1.5,
    # maximum acceleration in meters of line per second squared
    'MAX_ACCEL': 0.8,
    # sleep delay of tracking loop
    'LOOP_DELAY_S': 0.03,
    # record line length and tension every x iterations of tracking loop
    'REC_MOD': 3,
}

def constrain(value, minimum, maximum):
    return max(minimum, min(value, maximum))

class SpoolController:
    def __init__(self, motor, empty_diameter, full_diameter, full_length, conf, gear_ratio=1.0, tension_support=True):
        """
        Create a controller for a spool of line.

        empty_diameter_mm is the diameter of the spool in millimeters when no line is on it.
        full_diameter is the diameter in mm of the bulk of wrapped line when full_length meters of line are wrapped.
        line_capacity_m is the length of line in meters that is attached to this spool.
            if all of it were reeled in, the object at the end would reach the limit switch, if there is one.
        gear_ratio refers to how many rotations the spool makes for one rotation of the motor shaft.
        tension_support - whether the motor supports getShaftAngle which will be used to calculate tension.
            when true, assumes presens of anchor_server's conf vars
        """
        self.motor = motor
        self.empty_diameter = empty_diameter * 0.001 # millimeter to meters
        self.full_diameter = full_diameter * 0.001
        self.full_length = full_length
        self.gear_ratio = gear_ratio
        self.motor_orientation = -1
        self.tension_support = tension_support

        self.conf = conf
        self.conf.update(default_conf)

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
        # speed of line change in meters/sec last sent from controller.
        self.aim_line_speed = 0
        # whether to track a length plan 'plan' or use aim_line_speed 'speed'
        self.tracking_mode = 'plan'
        self.last_length = 3.0
        self.last_angle = 0.0
        self.meters_per_rev = self.get_unspool_rate(self.last_angle)
        # record of line length. tuples of (time, meters)
        self.record = []
        # plan of desired line lengths
        self.desired_line = []
        self.last_index = 0
        self.run_spool_loop = True
        self.rec_loop_counter = 0
        self.move_allowed = True
        self.abort_equalize_tension = False
        self.smoothed_tension = 0

        if self.tension_support:
            # the thresholds in the conf serve as a starting point, and the live values may change during tensioning.
            self.live_tension_low_thresh = self.conf['TENSION_SLACK_THRESH']
            self.live_tension_high_thresh = self.conf['TENSION_TIGHT_THRESH']

        # These two constants were stored in a file on their own before the broader conf dict was added
        # read the values in the file if they are present
        try:
            with open('mks42c_expected_err.cal', 'r') as f:
                self.conf['MKS42C_EXPECTED_ERR'] = float(f.readline())
                logging.info(f'Used stored mks42c_expected_err value of {self.conf["MKS42C_EXPECTED_ERR"]}')
                self.conf['MKS42C_TORQUE_FACTOR'] = float(f.readline())
                logging.info(f'Used stored mks42c_torque_factor value of {self.conf["MKS42C_TORQUE_FACTOR"]}')
        except (FileNotFoundError, ValueError):
            logging.warning('Could not read saved mks42c_expected_err')
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
            self.zero_angle = angle - relative_angle
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

    def _commandSpeed(self, speed):
        """command a specific speed from the motor."""
        self.speed = speed
        self.motor.runConstantSpeed(self.speed)

    def setPlan(self, plan):
        """
        Swith to plan tracking mode and set the plan to an array of tuples of time and length
        """
        self.tracking_mode = 'plan'
        self.desired_line = plan
        self.last_index = 0

    def setAimSpeed(self, lineSpeed):
        """Switch to speed tracking mode and set the aim speed in meters of line per second.
        negative values reel line in.
        """
        self.tracking_mode = 'speed'
        self.aim_line_speed = lineSpeed

    def jogRelativeLen(self, rel):
        new_l = self.last_length + rel
        self.setPlan([
            (time.time(), self.last_length),
            (time.time()+2, new_l),
        ])

    def popMeasurements(self):
        """Return up to DATA_LEN measurements. newest at the end."""
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
            return (time.time(), self.last_length)

        if abs(angle - self.last_angle) > 1:
            logging.warning(f'motor moved more than 1 rev since last read, last_angle={self.last_angle} angle={angle} diff={angle - self.last_angle}')
        self.last_angle = angle

        self.last_length = self.get_unspooled_length(angle)
        self.meters_per_rev = self.get_unspool_rate(angle)
        currentLineSpeed = self.speed * self.meters_per_rev

        self.move_allowed = True
        if self.last_length < 0 or self.last_length > self.full_length:
            logging.error(f"Bad length calculation! length={self.last_length}, shaftAngle={angle}. Movement disallowed until new reference length received.")
            # self.move_allowed = False

        if self.tension_support:
            success, tension = self.currentTension()
            if success:
                # because tension is never returned while stopped, this value only gets updated while moving.
                # clients can always look at it and expected a reasonable, if old, value
                fac = self.conf['TENSION_SMOOTHING_FACTOR']
                self.smoothed_tension = (tension * fac + self.smoothed_tension * (1-fac))

            if self.smoothed_tension > self.conf['DANGEROUS_TENSION']:
                logging.warning(f"Tension of {self.smoothed_tension} is too high!")
                # try to loosen up right away to avoid breaking something
                self._commandSpeed(1)
                time.sleep(1)
                self._commandSpeed(0)
                self.move_allowed = False
            row = (time.time(), self.last_length, currentLineSpeed, self.smoothed_tension)
        else:
            # accumulate these so you can send them to the websocket
            row = (time.time(), self.last_length, currentLineSpeed)

        if self.rec_loop_counter == self.conf['REC_MOD']:
            self.record.append(row)
            self.rec_loop_counter = 0
        self.rec_loop_counter += 1
        return time.time(), self.last_length

    def currentTension(self):
        """return the line tension in kilograms of force.
        Only designed for the MKS42C
        Measurements are basically invalid below speeds of 0.4
        """
        if self.speed < 0.3:
            return False, 0 # angle error readings are garbage at low or zero speeds.
        success, angleError = self.motor.getShaftError()
        if not success:
            return False, 0
        baseline = self.conf['MKS42C_EXPECTED_ERR'] * self.speed
        load_err = baseline - angleError # the residual position error attributable to load on the spool, not commanded motor speed
        # greater load on the line subtracts from angleError regardless of the commanded motor direction.
        torque = self.conf['MKS42C_TORQUE_FACTOR'] * load_err
        return True, torque / self.meters_per_rev

    def fastStop(self):
        # fast stop is permanent.
        # it causes the trackingloop task to stop,
        # causing the websocket connection to close
        self.motor.stop()
        self.run_spool_loop = False

    def pauseTrackingLoop(self):
        self.spoolPause = True

    def resumeTrackingLoop(self):
        self.spoolPause = False

    def getAimSpeedFromPlan(self, t):
        """Compute a speed to aim for based on our position in the length plan

        Combines speed error with positional error to get a speed to aim for (pre-enforcement of max accel)
        """
        targetLen = self.desired_line[self.last_index][1]
        position_err = targetLen - self.last_length
        if abs(position_err) < 0.001:
            return 0

        # What would the speed be between the two datapoints tha straddle the present?
        # Positive values mean line is lengthening
        # result is in meters of line per second
        # if there is only one desired length, we can't find two that straddle the present and have to use current length and time
        if self.last_index == 0:
            time_A = t
            len_A = self.last_length
        else:
            time_A = self.desired_line[self.last_index-1][0]
            len_A = self.desired_line[self.last_index-1][1]
        targetSpeed = ((self.desired_line[self.last_index][1] - len_A) / (self.desired_line[self.last_index][0] - time_A))
        speed_err = targetSpeed - self.speed * self.meters_per_rev

        # If our positional error was zero, we could go exactly that speed.
        # if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
        # (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
        # ideally we want to catch up in one step, but we have max acceleration constraints.
        return targetSpeed + position_err * self.conf['PE_TERM']; # meters of line per second

    def trackingLoop(self):
        """
        Constantly try to match the position and speed given in an array

        """
        while self.run_spool_loop:
            if self.spoolPause:
                time.sleep(0.2)
                continue
            try:
                t, currentLen = self.currentLineLength()

                # change in line length in meters per second
                if self.tracking_mode == 'plan':
                    # Find the earliest entry in desired_line that is still in the future.
                    while self.last_index < len(self.desired_line) and self.desired_line[self.last_index][0] <= t:
                        self.last_index += 1
                    # slow stop when there is no data to track
                    if self.last_index >= len(self.desired_line):
                        if abs(self.speed) > 0:
                            newspeed = self.speed * 0.9
                            if abs(newspeed) < 0.2:
                                newspeed = 0
                            logging.debug(f'Slow stopping {newspeed}')
                            self._commandSpeed(newspeed)
                        time.sleep(self.conf['LOOP_DELAY_S'])
                        continue
                    aimSpeed = self.getAimSpeedFromPlan(t)
                elif self.tracking_mode == 'speed':
                    aimSpeed = self.aim_line_speed
                else:
                    aimSpeed = 0
                print(f'aimSpeed {aimSpeed}')

                # limit the acceleration of the line
                currentSpeed = self.speed * self.meters_per_rev
                wouldAccel = (aimSpeed - currentSpeed) / self.conf['LOOP_DELAY_S']
                if wouldAccel > self.conf['MAX_ACCEL']:
                    aimSpeed = self.conf['MAX_ACCEL'] * self.conf['LOOP_DELAY_S'] + currentSpeed
                elif wouldAccel < -self.conf['MAX_ACCEL']:
                    aimSpeed = -self.conf['MAX_ACCEL'] * self.conf['LOOP_DELAY_S'] + currentSpeed

                maxspeed = self.motor.getMaxSpeed()

                if self.move_allowed:
                    cspeed = constrain(aimSpeed / self.meters_per_rev, -maxspeed, maxspeed)
                    # in revs per second.
                    if cspeed < 0.1:
                        cspeed = 0
                    self._commandSpeed(cspeed)
                else:
                    logging.warning(f"would move at speed={self.speed} but length is invalid. calibrate length first.")

                time.sleep(self.conf['LOOP_DELAY_S'])
            except serial.serialutil.SerialTimeoutException:
                logging.error('Lost serial contact with motor')
                break

    async def measureRefLoad(self, load=0, freq=5):
        """
        Obtain an estimate of the expected angle error with no load per commanded rev/sec specific to this motor.
        """
        self.pauseTrackingLoop()
        data = []
        for i in range(120):
            speed = math.sin(time.time()*freq)*3
            if abs(speed) < 0.4:
                if speed > 0:
                    speed = 0.4
                else:
                    speed = -0.4
            self._commandSpeed(speed)
            valid, err = self.motor.getShaftError()
            if valid:
                if load==0:
                    # with no load, we are measuring expected_err
                    data.append(err / speed)
                else:
                    # under load we are measuring torque_factor
                    # record what torque would be if torque_factor were 1
                    data.append((self.conf['MKS42C_EXPECTED_ERR'] * self.speed - err) / self.meters_per_rev)
            await asyncio.sleep(1/30)
        self._commandSpeed(0)
        logging.debug(f'Collected {len(data)} data points')
        if load==0:
            new_expected_err = sum(data)/len(data)
            # fun fact. if we were using degrees/sec to measure speed instead of revolutions/sec, the units of this constant would be seconds.
            # I guess it would mean how long in seconds the motor is from theoretically catching up to it's set point.
            # why is this different for every motor? it seems to change drastically if the motor's own calibration is performed with any mass
            # on the spool, so its probably just a function of the PID terms.
            logging.info(f'calibrated mks42c_expected_err = {new_expected_err} deg/(rev/sec)')
            self.conf['MKS42C_EXPECTED_ERR'] = new_expected_err
        else:
            # divide the actual load in kg by the mean measured torque value.
            new_torque_factor = load / (sum(data)/len(data))
            logging.info(f'calibrated mks42c_torque_factor = {new_torque_factor} kg/deg')
            self.conf['MKS42C_TORQUE_FACTOR'] = new_torque_factor
        # TODO send the new conf to the controller instead of writing it locally
        # with open('mks42c_expected_err.cal', 'w') as f:
        #     f.write(f'{self.mks42c_expected_err}\n')
        #     f.write(f'{self.mks42c_torque_factor}\n')
        self.resumeTrackingLoop()

    async def equalizeSpoolTension(self,
        controllerApprovalEvent,
        sendUpdatesFunc,
        maxLineChange=None,
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
        if maxLineChange is None:
            maxLineChange = self.conf['MAX_LINE_CHANGE_DURING_TEQ']
        self.pauseTrackingLoop()
        # make sure the tracking loop is definitely paused. it's on another thread. 
        await asyncio.sleep(self.conf['LOOP_DELAY_S'] * 2)
        self._commandSpeed(0)
        self.abort_equalize_tension = False
        t, curLength = self.currentLineLength()
        startLength = curLength
        line_delta = 0

        # reset thresholds to default
        self.live_tension_low_thresh = self.conf['TENSION_SLACK_THRESH']
        self.live_tension_high_thresh = self.conf['TENSION_TIGHT_THRESH']

        logging.info("Measuring tension in motion")
        # measuring tension at rest is not possible. measure in motion
        self._commandSpeed(self.conf['MEASUREMENT_SPEED'])
        for i in range(self.conf['MEASUREMENT_TICKS']):
            t, curLength = self.currentLineLength()
            logging.debug(f'getting stabilized tension reading {self.smoothed_tension}')
            await asyncio.sleep(1/30)
        # undo motion that occurred during reading
        self._commandSpeed(-self.conf['MEASUREMENT_SPEED'])
        await asyncio.sleep(self.conf['MEASUREMENT_TICKS']/30)
        self._commandSpeed(0)

        # decide initial speed. motor direction will not change during the loop
        if self.smoothed_tension < self.conf['TENSION_SLACK_THRESH']:
            started_slack = True
            self._commandSpeed(-self.conf['MOTOR_SPEED_DURING_TEQ'])
        else:
            started_slack = False
            self._commandSpeed(self.conf['MOTOR_SPEED_DURING_TEQ'])
        is_slack = started_slack

        logging.info(f'Started slack={started_slack} with a tension of {self.smoothed_tension} kg at a measurement speed of {self.conf["MEASUREMENT_SPEED"]} motor revs/s')
 
        try:
            # wait for stop condition
            while ((started_slack == is_slack)
                   and not self.abort_equalize_tension
                   and abs(line_delta) < maxLineChange
                   and (started_slack or allowOutSpooling)
                   and self.smoothed_tension < self.conf['DANGEROUS_TENSION']):
                await asyncio.sleep(1/30)
                # self.currentLineLength() causes length and tension to be calculated and recorded in a list that
                # is periodically flushed to the websocket by a task that is always running while the ws is connected
                t, curLength = self.currentLineLength()
                logging.debug(f'curLength={curLength} tension={self.smoothed_tension}')
                line_delta = curLength - startLength
                is_slack = self.smoothed_tension < self.live_tension_low_thresh
        except Exception as e:
            self._commandSpeed(0)
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
        self._commandSpeed(0)
        try:
            logging.info('Waiting for tension_eq result approval from controller')
            await asyncio.wait_for(controllerApprovalEvent.wait(), timeout=30)
            # the caller will have set this flag while we were waiting to indicate approval.
            if not self.abort_equalize_tension:
                self.setReferenceLength(startLength)
        except TimeoutError:
            logging.warning('Timed out Waiting for tension_eq result approval from controller')
        self.desired_line = []
        self.resumeTrackingLoop()
