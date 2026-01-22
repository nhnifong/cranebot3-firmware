import numpy as np
import time

class KalmanFilter:
    """A Kalman Filter to estimate a 3D position, velocity, and sensor biases.
    This version includes a dynamically calculated process noise covariance.
    """

    def __init__(self, sensor_names, acceleration_std_dev, bias_std_dev):
        """
        Initializes the Kalman Filter.

        Parameters:
            sensor_names (list): A list of strings for each sensor's name.
            acceleration_std_dev (float): Standard deviation of unmodeled acceleration.
                                          This is the primary tuning parameter for filter smoothness.
            bias_std_dev (float): Standard deviation of the sensor bias change.
        """
        # State vector: [x, y, z, vx, vy, vz, bias1_x, bias1_y, bias1_z, bias2_x, ...]^T
        self.sensor_names = sensor_names
        self.num_sensors = len(sensor_names)
        self.state_size = 6 + 3 * self.num_sensors
        self.state_estimate = np.zeros(self.state_size)
        self.state_covariance = np.eye(self.state_size)

        # Set initial values for the core state
        self.state_estimate[:6] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state_covariance[:6, :6] = np.diag([10.0] * 3 + [1.0] * 3)
        
        # Initialize bias states to zero with high uncertainty
        for i in range(self.num_sensors):
            start_idx = 6 + 3 * i
            self.state_covariance[start_idx:start_idx+3, start_idx:start_idx+3] = np.eye(3) * 10.0

        self.model_time = time.time()
        
        # The standard deviations for the process noise are stored here
        # and used to compute the process noise covariance dynamically in `predict`.
        self.acceleration_std_dev = acceleration_std_dev
        self.bias_std_dev = bias_std_dev
    
    def _get_F(self, delta_time):
        """Constructs the state transition matrix F for a given time step."""
        state_transition_matrix = np.eye(self.state_size)
        state_transition_matrix[0, 3] = delta_time
        state_transition_matrix[1, 4] = delta_time
        state_transition_matrix[2, 5] = delta_time
        return state_transition_matrix

    def _get_process_noise_covariance(self, delta_time):
        """
        Calculates the process noise covariance (Q) for a given time step.
        """
        q_accel_squared = self.acceleration_std_dev**2
        dt = delta_time
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        process_noise_covariance = np.zeros((self.state_size, self.state_size))

        # Fill in the 3x3 position/velocity blocks
        process_noise_covariance[:3, :3] = np.eye(3) * dt4 / 4 * q_accel_squared
        process_noise_covariance[:3, 3:6] = np.eye(3) * dt3 / 2 * q_accel_squared
        process_noise_covariance[3:6, :3] = np.eye(3) * dt3 / 2 * q_accel_squared
        process_noise_covariance[3:6, 3:6] = np.eye(3) * dt2 * q_accel_squared
        
        # Add the bias noise (assumed to be a random walk, constant Q)
        for i in range(self.num_sensors):
            start_idx = 6 + 3 * i
            process_noise_covariance[start_idx:start_idx+3, start_idx:start_idx+3] = np.eye(3) * self.bias_std_dev**2
        
        return process_noise_covariance

    def predict_present(self):
        return self.predict(time.time() - self.model_time)

    def predict(self, delta_time):
        """
        Predict the state and covariance forward in time.
        """
        state_transition_matrix = self._get_F(delta_time)
        process_noise_covariance = self._get_process_noise_covariance(delta_time)
        
        self.state_estimate = state_transition_matrix @ self.state_estimate
        self.state_covariance = state_transition_matrix @ self.state_covariance @ state_transition_matrix.T + process_noise_covariance
        self.model_time += delta_time

    def update(self, measurement_vector, measurement_time, sensor_noise_covariance, measurement_type, sensor_name=None):
        """
        Update the state estimate with a new measurement.
        measurement_type can be 'position' or 'velocity'.
        """
        assert(measurement_vector.shape == (3,))
        
        measurement_matrix = np.zeros((3, self.state_size))
        
        if measurement_type == 'position':
            assert sensor_name is not None, "sensor_name is required for position updates."
            sensor_idx = self.sensor_names.index(sensor_name)
            bias_start_idx = 6 + 3 * sensor_idx
            measurement_matrix[:3, :3] = np.eye(3)
            measurement_matrix[:3, bias_start_idx:bias_start_idx+3] = np.eye(3)
        
        elif measurement_type == 'velocity':
            measurement_matrix[:3, 3:6] = np.eye(3)

        else:
            raise ValueError("Invalid measurement_type. Must be 'position' or 'velocity'.")
            
        # Retrodict (propagate backwards) from current state to measurement time
        delta_time = self.model_time - measurement_time
        state_transition_matrix_retro = self._get_F(-delta_time)
        
        state_at_meas_time = state_transition_matrix_retro @ self.state_estimate
        cov_at_meas_time = state_transition_matrix_retro @ self.state_covariance @ state_transition_matrix_retro.T
        
        # Perform standard update step at the measurement time
        innovation = measurement_vector - measurement_matrix @ state_at_meas_time
        innovation_covariance = measurement_matrix @ cov_at_meas_time @ measurement_matrix.T + sensor_noise_covariance
        kalman_gain = cov_at_meas_time @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        
        corrected_state_at_meas_time = state_at_meas_time + kalman_gain @ innovation
        corrected_cov_at_meas_time = (np.eye(self.state_size) - kalman_gain @ measurement_matrix) @ cov_at_meas_time
        
        # Propagate corrected state forward to the current time
        prop_F = self._get_F(delta_time)
        prop_Q = self._get_process_noise_covariance(delta_time)
        
        self.state_estimate = prop_F @ corrected_state_at_meas_time
        self.state_covariance = prop_F @ corrected_cov_at_meas_time @ prop_F.T + prop_Q

    def reset_biases(self, perfect_position):
        """
        Resets the biases and state estimate by performing a perfect update.
        This is useful for re-calibrating the system.
        """
        # Define a measurement matrix for a perfect position sensor
        perfect_measurement_matrix = np.zeros((3, self.state_size))
        perfect_measurement_matrix[:3, :3] = np.eye(3)
        
        # A perfect measurement has zero covariance
        perfect_measurement_covariance = np.zeros((3, 3))
        
        # Perform a standard update
        innovation = perfect_position - perfect_measurement_matrix @ self.state_estimate
        innovation_covariance = perfect_measurement_matrix @ self.state_covariance @ perfect_measurement_matrix.T + perfect_measurement_covariance
        
        # The Kalman gain will be large due to zero covariance, effectively resetting the state
        kalman_gain = self.state_covariance @ perfect_measurement_matrix.T @ np.linalg.inv(innovation_covariance + np.eye(3) * 1e-9) # Add a small value to prevent singular matrix
        
        self.state_estimate = self.state_estimate + kalman_gain @ innovation
        self.state_covariance = (np.eye(self.state_size) - kalman_gain @ perfect_measurement_matrix) @ self.state_covariance

    def enforce_bias_constraint(self, constraint_strength=0.01):
        """
        Applies a pseudo-measurement to enforce that the sum of all sensor biases
        is approximately zero.
        """
        # A pseudo-measurement of zero
        measurement_vector = np.zeros(3)
        
        # The measurement noise covariance matrix for the constraint.
        # A smaller value means a stronger constraint.
        sensor_noise_covariance = np.eye(3) * constraint_strength

        # Construct the measurement matrix H for the bias sum constraint
        measurement_matrix = np.zeros((3, self.state_size))
        for i in range(self.num_sensors):
            # The columns corresponding to each sensor's bias
            bias_start_idx = 6 + 3 * i
            measurement_matrix[:3, bias_start_idx:bias_start_idx+3] = np.eye(3)

        # Perform the standard Kalman update using this pseudo-measurement
        # Use the current state and covariance, since this is an immediate update
        innovation = measurement_vector - measurement_matrix @ self.state_estimate
        innovation_covariance = measurement_matrix @ self.state_covariance @ measurement_matrix.T + sensor_noise_covariance
        
        kalman_gain = self.state_covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        
        self.state_estimate = self.state_estimate + kalman_gain @ innovation
        self.state_covariance = (np.eye(self.state_size) - kalman_gain @ measurement_matrix) @ self.state_covariance
