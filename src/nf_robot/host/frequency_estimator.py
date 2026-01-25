import collections
import time
import math
from random import random
from scipy.spatial.transform import Rotation

class FrequencyEstimator:
    """
    Estimates the frequency of a single low-frequency signal from timestamped
    measurements using an efficient "online" algorithm.

    it does not store a window of historical data.
    It processes measurements as they arrive and uses an
    exponential moving average to smooth the frequency estimate for stability.
    """

    def __init__(self, hysteresis=0.1, smoothing_factor=0.1):
        """
        Initializes the FrequencyEstimator.

        Args:
            hysteresis (float): The threshold for zero-crossing detection to make
                it robust to noise.
            smoothing_factor (float): The alpha for the exponential moving average
                used to smooth the frequency estimate. A smaller value means
                more smoothing but slower response to frequency changes.
        """
        self._min_freq = 0.2  # Hz
        self._max_freq = 6.0   # Hz
        self._hysteresis = hysteresis
        self._alpha = smoothing_factor

        # State variables for the online algorithm
        self._state = 0  # -1: below low, 0: undefined, 1: above high
        self._last_timestamp = None
        self._last_value = None
        self._last_crossing_time = None
        self._smoothed_frequency = None

    def add_measurement(self, timestamp: float, value: float):
        """
        Adds and processes a new timestamped measurement.
        This method updates the internal state and frequency estimate.

        Args:
            timestamp (float): The Unix timestamp of the measurement.
            value (float): The measurement value.
        """
        if self._last_value is None:
            self._last_timestamp = timestamp
            self._last_value = value
            return

        high_threshold = self._hysteresis
        low_threshold = -self._hysteresis

        # Initialize state if it's undefined
        if self._state == 0:
            if self._last_value > high_threshold:
                self._state = 1
            elif self._last_value < low_threshold:
                self._state = -1

        # --- Detect crossing and update frequency ---
        crossing_detected = False
        target_threshold = 0

        # Detect a rising crossing (from low to high)
        if self._state == -1 and value > high_threshold and self._last_value <= high_threshold:
            self._state = 1
            crossing_detected = True
            target_threshold = high_threshold
        # Detect a falling crossing (from high to low)
        elif self._state == 1 and value < low_threshold and self._last_value >= low_threshold:
            self._state = -1
            crossing_detected = True
            target_threshold = low_threshold

        if crossing_detected:
            # Interpolate to find a more precise crossing time
            t1, v1 = self._last_timestamp, self._last_value
            t2, v2 = timestamp, value
            interpolated_time = t1 + (t2 - t1) * (target_threshold - v1) / (v2 - v1)

            if self._last_crossing_time is not None:
                half_period = interpolated_time - self._last_crossing_time
                if half_period > 0:
                    # Calculate instantaneous frequency
                    period = 2 * half_period
                    frequency = 1.0 / period
                    
                    # Clamp frequency to the expected range
                    frequency = max(self._min_freq, min(self._max_freq, frequency))

                    # Apply exponential moving average for smoothing
                    if self._smoothed_frequency is None:
                        self._smoothed_frequency = frequency
                    else:
                        self._smoothed_frequency = (self._alpha * frequency) + \
                                                   (1 - self._alpha) * self._smoothed_frequency

            self._last_crossing_time = interpolated_time

        # Update last known values
        self._last_timestamp = timestamp
        self._last_value = value

    def get_frequency(self) -> float | None:
        """
        Returns the current smoothed frequency estimate.

        Returns:
            float or None: The estimated frequency in Hz, or None if a reliable
                           estimate has not yet been established.
        """
        return self._smoothed_frequency

class SwingFrequencyEstimator:
    """
    Wraps two FrequencyEstimator instances to robustly estimate the swing
    frequency of an IMU-equipped object (like a gripper) that can both
    swing (pendulum motion) and spin (rotate around its Z-axis).

    This class decouples the swing from the spin, which is crucial for
    an accurate estimate.
    """
    def __init__(self, hysteresis=0.1, smoothing_factor=0.05):
        """
        Initializes the swing frequency estimator.

        Args:
            hysteresis (float): The hysteresis for zero-crossing detection,
                passed to the underlying estimators. This should be based on
                the noise level of the calculated swing components (zx, zy).
                Since these are unit vector components, a value like 0.02
                might be more appropriate than 0.1. Let's make it 0.05.
            smoothing_factor (float): The smoothing factor for the EMA,
                passed to the underlying estimators.
        """
        # We use two estimators: one for the swing component in the world's
        # X-direction and one for the Y-direction. This makes the system
        # robust to swings in any direction (planar, elliptical, circular).
        self._estimator_x = FrequencyEstimator(
            hysteresis=hysteresis, 
            smoothing_factor=smoothing_factor
        )
        self._estimator_y = FrequencyEstimator(
            hysteresis=hysteresis, 
            smoothing_factor=smoothing_factor
        )
        
        # The IMU's Z-axis (up) vector in its local frame
        self._z_imu_local = [0, 0, 1]

    def add_rotation_vector(self, timestamp: float, rotvec: list[float]):
        """
        Adds and processes a new timestamped rotation vector.
        
        This method calculates the pure swing components and feeds them
        to the internal frequency estimators.

        Args:
            timestamp (float): The Unix timestamp of the measurement.
            rotvec (list[float]): The rotation vector [rx, ry, rz] from the IMU.
        """
        r = Rotation.from_rotvec(rotvec)
        z_world = r.apply(self._z_imu_local)
        zx = z_world[0]
        zy = z_world[1]
        self._estimator_x.add_measurement(timestamp, zx)
        self._estimator_y.add_measurement(timestamp, zy)

    def get_frequency(self) -> float | None:
        """
        Returns the combined, robust frequency estimate.

        It checks the estimates from both the X and Y components and
        averages them if both are valid, or returns whichever one is valid.
        
        Returns:
            float or None: The estimated swing frequency in Hz, or None.
        """
        freq_x = self._estimator_x.get_frequency()
        freq_y = self._estimator_y.get_frequency()

        if freq_x is not None and freq_y is not None:
            # Both are valid, return the average. They should be very close.
            return (freq_x + freq_y) / 2.0
        elif freq_x is not None:
            # Only X-swing is detected
            return freq_x
        elif freq_y is not None:
            # Only Y-swing is detected
            return freq_y
        else:
            # No stable frequency detected yet
            return None

    def get_pendulum_length(self, g: float = 9.80665) -> float | None:
        """
        Calculates the effective length of the pendulum based on the
        current estimated swing frequency.

        Returns:
            float | None: The estimated pendulum length in meters, or None
                          if the frequency cannot be determined.
        """
        frequency = self.get_frequency()
        if frequency is None or frequency <= 0:
            return None
            
        # L = g / (4 * pi^2 * f^2)
        try:
            length = g / (4.0 * (math.pi ** 2) * (frequency ** 2))
            return length
        except (OverflowError, ValueError):
            return None


if __name__ == '__main__':
    # --- Example Usage ---

    # Instantiate the new wrapper class
    # Note: Hysteresis is now applied to the 'zx' and 'zy' components,
    # which range from approx -0.3 to +0.3 in this sim, so 0.05 is reasonable.
    swing_estimator = SwingFrequencyEstimator(hysteresis=0.05, smoothing_factor=0.05)

    # Simulate a signal
    true_swing_freq = 1.9  # Hz
    true_spin_freq = 0.7   # Hz (Gripper is also spinning)
    
    swing_amplitude_x = 0.2  # radians (swing in X)
    swing_amplitude_y = 0.3  # radians (swing in Y, creating an elliptical path)

    sampling_rate = 30     # Measurements per second
    simulation_duration = 15 # seconds

    print(f"Simulating a {true_swing_freq} Hz swing and a {true_spin_freq} Hz spin...")
    print("-" * 30)

    start_time = time.time()
    current_time = start_time
    
    last_print_time = 0

    while current_time - start_time < simulation_duration:
        timestamp = time.time()
        elapsed = timestamp - start_time
        
        # Generate the simulated motion
        
        # Swing motion (pendulum)
        swing_x = swing_amplitude_x * math.sin(elapsed * true_swing_freq * 2 * math.pi)
        swing_y = swing_amplitude_y * math.cos(elapsed * true_swing_freq * 2 * math.pi)
        
        # Spin motion (rotation around Z)
        spin_z = elapsed * true_spin_freq * 2 * math.pi
        
        # Combine rotations to get the total orientation
        # We use intrinsic 'zyx' order: spin first, then tilt.
        # This simulates the gripper spinning and tilting.
        r_total = Rotation.from_euler('zyx', [spin_z, swing_y, swing_x])
        
        # Get the rotation vector (this is what the IMU provides)
        rotvec = r_total.as_rotvec()
        
        # Add some noise to the rotation vector
        rotvec += [0.01 * (random() - 0.5) for _ in range(3)]

        # Add and process the measurement
        swing_estimator.add_rotation_vector(timestamp, rotvec)
        
        # Get the current frequency estimate
        if (current_time - last_print_time) > 1.0:
            estimated_freq = swing_estimator.get_frequency()
            if estimated_freq is not None:
                print(f"Time: {elapsed:.1f}s | Estimated SWING Freq: {estimated_freq:.3f} Hz")
            last_print_time = current_time