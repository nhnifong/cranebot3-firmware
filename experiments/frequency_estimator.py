import collections
import time
import math
from random import random

class FrequencyEstimator:
    """
    Estimates the frequency of a single low-frequency signal from timestamped
    measurements.

    This class is designed to be efficient for real-time applications where
    measurements arrive at a variable rate. It uses a sliding window of data
    and calculates frequency based on the timing of zero-crossings.
    """

    def __init__(self, window_size_seconds=10, expected_rate=30, hysteresis=0.1):
        """
        Initializes the FrequencyEstimator.

        Args:
            window_size_seconds (int): The duration of the sliding window
                for storing measurements, in seconds. This should be long
                enough to capture several cycles of the lowest expected frequency.
            expected_rate (int): The expected mean measurement rate (per second).
                This is used to set the size of the internal data buffer.
            hysteresis (float): The threshold for zero-crossing detection to make
                it robust to noise. A crossing is detected when the signal
                passes from below -hysteresis to above +hysteresis, or vice versa.
        """
        # Calculate the maximum number of samples to store based on the window size
        # and expected rate. A buffer factor of 1.5 is added for safety.
        max_samples = int(window_size_seconds * expected_rate * 1.5)

        # Use collections.deque for efficient, fixed-size storage (sliding window)
        self.timestamps = collections.deque(maxlen=max_samples)
        self.values = collections.deque(maxlen=max_samples)
        self._min_freq = 0.2  # Hz
        self._max_freq = 6.0   # Hz
        self._hysteresis = hysteresis

    def add_measurement(self, timestamp: float, value: float):
        """
        Adds a new timestamped measurement to the estimator.

        Args:
            timestamp (float): The Unix timestamp of the measurement.
            value (float): The measurement value, expected to be between -1 and 1.
        """
        # if not (-1.0 <= value <= 1.0):
        #     raise ValueError("Measurement value must be between -1 and 1.")
            
        self.timestamps.append(timestamp)
        self.values.append(value)

    def get_frequency(self) -> float | None:
        """
        Calculates the current estimated frequency of the signal.

        Returns:
            float or None: The estimated frequency in Hz, or None if a reliable
                           estimate cannot be made (e.g., not enough data or
                           no zero-crossings found).
        """
        if len(self.values) < 2:
            return None # Not enough data to find any crossings

        crossing_times = []
        
        # State: 0 = undefined, 1 = above high threshold, -1 = below low threshold
        state = 0 
        
        # Define high and low thresholds for hysteresis
        high_threshold = self._hysteresis
        low_threshold = -self._hysteresis

        for i in range(1, len(self.values)):
            v_prev = self.values[i-1]
            v_curr = self.values[i]

            # Initialize state with the first definite position
            if state == 0:
                if v_prev > high_threshold:
                    state = 1
                elif v_prev < low_threshold:
                    state = -1
                continue

            # Detect a rising crossing (from low to high)
            if state == -1 and v_curr > high_threshold and v_prev <= high_threshold:
                t1, v1 = self.timestamps[i-1], self.values[i-1]
                t2, v2 = self.timestamps[i], self.values[i]
                # Interpolate to find time of crossing the high threshold
                if v2 != v1:
                    interpolated_time = t1 + (t2 - t1) * (high_threshold - v1) / (v2 - v1)
                    crossing_times.append(interpolated_time)
                state = 1
            # Detect a falling crossing (from high to low)
            elif state == 1 and v_curr < low_threshold and v_prev >= low_threshold:
                t1, v1 = self.timestamps[i-1], self.values[i-1]
                t2, v2 = self.timestamps[i], self.values[i]
                # Interpolate to find time of crossing the low threshold
                if v2 != v1:
                    interpolated_time = t1 + (t2 - t1) * (low_threshold - v1) / (v2 - v1)
                    crossing_times.append(interpolated_time)
                state = -1

        if len(crossing_times) < 2:
            return None # Need at least two crossings to measure a period

        # Calculate the time differences between consecutive crossings
        # These represent the duration of half-cycles of the signal
        half_periods = [crossing_times[i] - crossing_times[i-1] for i in range(1, len(crossing_times))]

        if not half_periods:
            return None

        # Calculate the average half-period
        avg_half_period = sum(half_periods) / len(half_periods)
        
        # Avoid division by zero
        if avg_half_period == 0:
            return None

        # The full period is twice the half-period
        period = 2 * avg_half_period
        
        # Frequency is the reciprocal of the period
        frequency = 1.0 / period

        # Clamp the frequency to the expected range to filter out noise
        return max(self._min_freq, min(self._max_freq, frequency))

if __name__ == '__main__':
    # --- Example Usage ---
    
    # Instantiate the estimator with hysteresis to handle noise
    estimator = FrequencyEstimator(hysteresis=0.2)

    # Simulate a signal
    true_frequency = 0.5  # Hz
    sampling_rate = 30    # Measurements per second
    simulation_duration = 15 # seconds

    print(f"Simulating a {true_frequency} Hz signal for {simulation_duration} seconds...")
    print("-" * 30)

    start_time = time.time()
    current_time = start_time
    
    while current_time - start_time < simulation_duration:
        # Generate a measurement
        timestamp = time.time() + (random()-0.5)/10
        # Add more significant noise that crosses the zero-line
        value = math.sin(timestamp * true_frequency * 2 * math.pi) + (random()-0.5) * 0.4 
        
        # Add the measurement to the estimator
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


