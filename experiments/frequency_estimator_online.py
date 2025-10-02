import collections
import time
import math
from random import random

class FrequencyEstimator:
    """
    Estimates the frequency of a single low-frequency signal from timestamped
    measurements using an efficient "online" algorithm.

    This version is highly memory-efficient as it does not store a window of
    historical data. It processes measurements as they arrive and uses an
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

if __name__ == '__main__':
    # --- Example Usage ---

    # Instantiate the estimator. A lower smoothing factor means more smoothing.
    estimator = FrequencyEstimator(hysteresis=0.2, smoothing_factor=0.05)

    # Simulate a signal
    true_frequency = 1.9  # Hz
    sampling_rate = 30    # Measurements per second
    simulation_duration = 15 # seconds

    print(f"Simulating a {true_frequency} Hz signal for {simulation_duration} seconds...")
    print("-" * 30)

    start_time = time.time()
    current_time = start_time
    
    while current_time - start_time < simulation_duration:
        # Generate a measurement
        timestamp = time.time() + (random()-0.5)/10
        # Add significant noise that crosses the zero-line
        value = math.sin(timestamp * true_frequency * 2 * math.pi) + (random()-0.5) * 0.4 
        
        # Add and process the measurement
        estimator.add_measurement(timestamp, value)
        
        # Get the current frequency estimate every second
        if int(current_time) > int(start_time) and int(current_time) != int(current_time - (1.0/sampling_rate)):
             estimated_freq = estimator.get_frequency()
             if estimated_freq is not None:
                 print(f"Time: {current_time - start_time:.1f}s | Estimated Frequency: {estimated_freq:.3f} Hz")
        
        # Wait to simulate the sampling rate
        time.sleep(1.0 / sampling_rate)
        current_time = time.time()

    print("-" * 30)
    final_estimate = estimator.get_frequency()
    if final_estimate is not None:
        print(f"Final Estimated Frequency: {final_estimate:.3f} Hz")
        print(f"True Frequency: {true_frequency} Hz")
        print(f"Error: {abs(final_estimate - true_frequency):.3f} Hz")
    else:
        print("Could not determine a final frequency estimate.")

