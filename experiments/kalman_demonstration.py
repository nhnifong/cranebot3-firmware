import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import ceil

# --- Sensor and Data Classes ---

class Measurement:
    """A class to hold a sensor measurement with its timestamp and value."""
    def __init__(self, timestamp, position, sensor):
        self.timestamp = timestamp
        self.position = position
        self.sensor = sensor # Now holds a reference to the sensor object

class Sensor:
    """A class to represent a sensor with its specific properties."""
    def __init__(self, name, latency, noise_std_dev, frequency=0.5):
        self.name = name
        self.latency = latency  # seconds
        self.frequency = frequency
        self.noise_std_dev = noise_std_dev
        self.noise_covariance = np.diag([noise_std_dev**2] * 3)
        self.last_measurement_time = 0.0

# --- Kalman Filter Implementation ---

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
        
        # Measurement matrix: maps state to measurement (position only)
        self.measurement_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        self.last_update_time = 0.0
        self.state_history = [(self.state_estimate.copy(), self.state_covariance.copy(), 0.0)]

    def predict(self, delta_time):
        """Predict the state and covariance forward in time."""
        self.state_transition_matrix[0, 3] = delta_time
        self.state_transition_matrix[1, 4] = delta_time
        self.state_transition_matrix[2, 5] = delta_time

        self.state_estimate = self.state_transition_matrix @ self.state_estimate
        self.state_covariance = self.state_transition_matrix @ self.state_covariance @ self.state_transition_matrix.T + self.process_noise_covariance

    def update(self, measurement, current_time):
        """Update the state estimate with a new measurement and its timestamp."""
        
        # Find the state estimate at the time the measurement was taken
        measurement_time = current_time - measurement.sensor.latency
        
        # Retrodict (propagate backwards) from current state to measurement time
        delta_time_retrodict = current_time - measurement_time
        
        # State and covariance at the measurement timestamp
        state_at_meas_time = self.state_estimate.copy()
        cov_at_meas_time = self.state_covariance.copy()
        
        # Propagate backward (invert F)
        F_inv = np.linalg.inv(self.state_transition_matrix)
        for _ in range(int(delta_time_retrodict / 0.025)): # Assuming 0.025 is the smallest dt
            state_at_meas_time = F_inv @ state_at_meas_time
            cov_at_meas_time = F_inv @ cov_at_meas_time @ F_inv.T
            
        # Perform standard update step at the measurement time
        innovation = measurement.position - self.measurement_matrix @ state_at_meas_time
        innovation_covariance = self.measurement_matrix @ cov_at_meas_time @ self.measurement_matrix.T + measurement.sensor.noise_covariance
        kalman_gain = cov_at_meas_time @ self.measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        
        corrected_state_at_meas_time = state_at_meas_time + kalman_gain @ innovation
        corrected_cov_at_meas_time = (np.eye(6) - kalman_gain @ self.measurement_matrix) @ cov_at_meas_time
        
        # Propagate corrected state forward to the current time
        delta_time_propagate = current_time - measurement_time
        prop_F = np.eye(6)
        prop_F[0, 3] = delta_time_propagate
        prop_F[1, 4] = delta_time_propagate
        prop_F[2, 5] = delta_time_propagate
        
        self.state_estimate = prop_F @ corrected_state_at_meas_time
        self.state_covariance = prop_F @ corrected_cov_at_meas_time @ prop_F.T + self.process_noise_covariance
        
        self.last_update_time = current_time
        

# --- Simulation Logic ---

def simulate_system(duration, time_step, sensors):
    """Simulates the system and runs the Kalman filter."""
    
    # Initialize the true state of the object with floats
    true_state = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])  # [x, y, z, vx, vy, vz]
    
    # Initialize the Kalman filter with floats
    initial_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_covariance = np.diag([10.0] * 3 + [1.0] * 3)
    kf = KalmanFilter(initial_estimate, initial_covariance)

    # Use a queue to hold incoming measurements
    measurement_queue = deque()
    
    true_path_history = []
    estimated_path_history = []
    
    # Store measurements for plotting
    measurements_a = []
    measurements_b = []
    measurements_c = []

    # Simulation loop
    for step in range(int(duration / time_step)):
        current_time = step * time_step
        
        # Predict the state forward in time
        kf.predict(time_step)
        
        # Simulate true object motion with occasional trajectory changes
        true_state[0:3] += true_state[3:6] * time_step
        if step > 0 and step % 50 == 0:
            true_state[3:6] = np.zeros(3) # Stop
        if step > 0 and step % 100 == 0:
            true_state[3:6] = np.random.randn(3) * 1.5 # Change velocity
            
        # Simulate measurements from all sensors and add them to the queue
        for sensor in sensors:
            if current_time - sensor.last_measurement_time >= sensor.frequency:
                # The true position at the time the measurement was taken
                true_position_at_measurement = true_state[:3] - true_state[3:6] * sensor.latency
                measured_position = true_position_at_measurement + np.random.multivariate_normal([0, 0, 0], sensor.noise_covariance)
                measurement_queue.append(Measurement(current_time, measured_position, sensor))
                sensor.last_measurement_time = current_time

        # Process all measurements that are now "current"
        while measurement_queue:
            measurement = measurement_queue.popleft()
            
            # Store the measurement for plotting
            if measurement.sensor.name.startswith('Sensor A'):
                measurements_a.append(measurement.position)
            elif measurement.sensor.name.startswith('Sensor B'):
                measurements_b.append(measurement.position)
            elif measurement.sensor.name.startswith('Sensor C'):
                measurements_c.append(measurement.position)
            
            # Use the sensor object reference from the measurement to get the correct covariance
            kf.update(measurement, current_time)
            
        # Store the history for plotting
        true_path_history.append(true_state[:3].copy())
        estimated_path_history.append(kf.state_estimate[:3].copy())
            
    return np.array(true_path_history), np.array(estimated_path_history), np.array(measurements_a), np.array(measurements_b), np.array(measurements_c)

# --- Main Execution ---

if __name__ == "__main__":
    
    # Define the sensors
    sensors = [
        Sensor(name="Sensor A (Low Latency, Low Acc)", latency=0.05, noise_std_dev=0.3, frequency=0.1),
        Sensor(name="Sensor B (Med Latency, Med Acc)", latency=0.5, noise_std_dev=0.3),
        Sensor(name="Sensor C (High Latency, High Acc)", latency=2.0, noise_std_dev=0.06)
    ]

    # Run the simulation
    duration = 20.0
    time_step = 0.025 # A small time step for prediction
    true_path, estimated_path, measurements_a, measurements_b, measurements_c = simulate_system(duration, time_step, sensors)

    # Plotting the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2], label='True Path')
    ax.plot(estimated_path[:, 0], estimated_path[:, 1], estimated_path[:, 2], label='Kalman Filter Estimate', linestyle='--', linewidth=2)
    
    # Plotting the individual measurements
    if len(measurements_a) > 0:
        ax.scatter(measurements_a[:, 0], measurements_a[:, 1], measurements_a[:, 2], c='orange', marker='o', label='Sensor A Measurements', alpha=0.6)
    if len(measurements_b) > 0:
        ax.scatter(measurements_b[:, 0], measurements_b[:, 1], measurements_b[:, 2], c='green', marker='^', label='Sensor B Measurements', alpha=0.8)
    if len(measurements_c) > 0:
        ax.scatter(measurements_c[:, 0], measurements_c[:, 1], measurements_c[:, 2], c='red', marker='s', label='Sensor C Measurements', alpha=1.0)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Kalman Filter with Multi-Sensor Fusion (Delayed Data)')
    ax.legend()
    plt.show()