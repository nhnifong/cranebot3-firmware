import numpy as np
import time

class KalmanFilter:
    """A Kalman Filter to estimate a 3D position and velocity state."""
    def __init__(self, initial_state, initial_covariance):
        # State vector: [x, y, z, vx, vy, vz]^T
        self.state_estimate = initial_state
        # State covariance matrix
        self.state_covariance = initial_covariance
        
        # We assume a constant velocity model
        self.state_transition_matrix = np.eye(6)
        
        # Process noise accounts for uncertainty in our model
        self.process_noise_covariance = np.diag([0.01, 0.01, 0.01, 0.005, 0.005, 0.05])
        
        # Measurement matrix: maps state to measurement (we can measure position only)
        self.measurement_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        self.model_time = time.time()

    def predict_present(self):
        return self.predict(time.time() - self.model_time)

    def predict(self, delta_time):
        """Predict the state and covariance forward in time."""
        self.state_transition_matrix[0, 3] = delta_time
        self.state_transition_matrix[1, 4] = delta_time
        self.state_transition_matrix[2, 5] = delta_time

        self.state_estimate = self.state_transition_matrix @ self.state_estimate
        self.state_covariance = self.state_transition_matrix @ self.state_covariance @ self.state_transition_matrix.T + self.process_noise_covariance

        self.model_time += delta_time

    def update(self, position, measurement_time, sensor_noise_covariance):
        """Update the state estimate with a new measurement and its timestamp."""
        
        # Retrodict (propagate backwards) from current state to measurement time
        delta_time = self.model_time - measurement_time
        
        # Retro-propagation state transition matrix (F_inv_exact)
        # Note: In a constant velocity model, this is the same as F with a negative time component.
        F_retro = np.eye(6)
        F_retro[0, 3] = -delta_time
        F_retro[1, 4] = -delta_time
        F_retro[2, 5] = -delta_time

        # State and covariance at the measurement timestamp
        state_at_meas_time = F_retro @ self.state_estimate
        cov_at_meas_time = F_retro @ self.state_covariance @ F_retro.T
        
        # Perform standard update step at the measurement time
        innovation = position - self.measurement_matrix @ state_at_meas_time
        innovation_covariance = self.measurement_matrix @ cov_at_meas_time @ self.measurement_matrix.T + sensor_noise_covariance
        kalman_gain = cov_at_meas_time @ self.measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        
        corrected_state_at_meas_time = state_at_meas_time + kalman_gain @ innovation
        corrected_cov_at_meas_time = (np.eye(6) - kalman_gain @ self.measurement_matrix) @ cov_at_meas_time
        
        # Propagate corrected state forward to the current time
        prop_F = np.eye(6)
        prop_F[0, 3] = delta_time
        prop_F[1, 4] = delta_time
        prop_F[2, 5] = delta_time
        
        self.state_estimate = prop_F @ corrected_state_at_meas_time
        self.state_covariance = prop_F @ corrected_cov_at_meas_time @ prop_F.T + self.process_noise_covariance