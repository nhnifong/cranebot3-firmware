from dataclasses import dataclass, field
import numpy as np
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep
from lerobot.teleoperators.phone.config_phone import PhoneOS
import time
from scipy.spatial.transform import Rotation


# --- Tunable Constants ---
# Feel free to adjust these gains to make the robot more or less sensitive.
# Scales phone velocity (in m/s) to gantry velocity (in m/s).
GANTRY_VEL_GAIN = 1.2
# The winch speed (in m/s) when a button is held.
WINCH_SPEED_CONSTANT = 0.06
# Smoothing factor for the velocity calculation (0 < alpha <= 1).
# A smaller value means more smoothing but more latency.
VELOCITY_SMOOTHING_ALPHA = 0.6


@ProcessorStepRegistry.register("map_phone_action_to_stringman_action")
@dataclass
class MapPhoneActionToStringmanAction(RobotActionProcessorStep):
    """
    Maps calibrated phone pose actions to Stringman robot actions.

    This processor step converts the 6-DoF pose from the phone teleoperator
    into velocity commands for the Stringman robot's gantry, winch, and fingers.

    Control Mapping:
    - Gantry Velocity: Proportional to the phone's calculated 3D velocity.
    - Winch Speed: Controlled by dedicated buttons on the phone.
    - Finger Angle: Directly mapped from the phone's roll (twist) angle.

    Attributes:
        platform: The operating system of the phone (iOS or Android), used
            to determine the correct button mappings for the winch.
    """

    platform: PhoneOS
    # Private fields to store state for velocity calculation
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_time: float | None = field(default=None, init=False, repr=False)
    _smoothed_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        """
        Processes the phone action dictionary to create a Stringman action dictionary.

        Args:
            action: The input action dictionary from the phone teleoperator.

        Returns:
            A new action dictionary formatted for the Stringman robot controller.
        """
        # Pop the phone-specific keys from the action dictionary
        enabled = bool(action.pop("phone.enabled"))
        pos = action.pop("phone.pos")
        rot = action.pop("phone.rot")
        inputs = action.pop("phone.raw_inputs")

        # Calculate and Smooth Phone Velocity
        current_time = time.time()
        velocity = np.zeros(3)

        if self._last_pos is not None and self._last_time is not None:
            dt = current_time - self._last_time
            if dt > 1e-5:  # Avoid division by zero
                # Calculate raw velocity
                velocity = (pos - self._last_pos) / dt

        # Apply exponential moving average for smoothing
        self._smoothed_velocity = (
            VELOCITY_SMOOTHING_ALPHA * velocity + (1 - VELOCITY_SMOOTHING_ALPHA) * self._smoothed_velocity
        )

        # Update state for the next step
        self._last_pos = pos
        self._last_time = current_time

        # If teleop is disabled, reset velocity to prevent lingering movement
        if not enabled:
            self._smoothed_velocity = np.zeros(3)

        # Map Smoothed Velocity to Gantry Velocity
        gantry_vel_x = -self._smoothed_velocity[0] * GANTRY_VEL_GAIN
        gantry_vel_y = self._smoothed_velocity[1] * GANTRY_VEL_GAIN
        gantry_vel_z = self._smoothed_velocity[2] * GANTRY_VEL_GAIN

        # Map Buttons to Winch Speed ---
        winch_line_speed = 0.0
        if self.platform == PhoneOS.IOS:
            # Using buttons B2 and B3 for winch up/down on iOS
            winch_up = float(inputs.get("b2", 0.0))
            winch_down = float(inputs.get("b3", 0.0))
            winch_line_speed = WINCH_SPEED_CONSTANT * (winch_down - winch_up)
        else:  # Android
            # Using the reserved buttons A and B for winch up/down
            winch_up = float(inputs.get("reservedButtonA", 0.0))
            winch_down = float(inputs.get("reservedButtonB", 0.0))
            winch_line_speed = WINCH_SPEED_CONSTANT * (winch_down - winch_up)

        # Map Phone Roll to Finger Angle 
        # Get the phone's orientation in Euler angles (roll, pitch, yaw).
        # We use the 'xyz' sequence, where the first angle ('x') corresponds to roll.
        scipy_rot = Rotation.from_quat(rot.as_quat())
        roll, _pitch, _yaw = scipy_rot.as_euler("xyz", degrees=True)

        # Clamp the angle to the robot's valid range [-90, 90].
        # The finger angle is sent even when disabled to allow independent control.
        finger_angle = np.clip(roll, -90, 90)

        action["gantry_vel_x"] = gantry_vel_x
        action["gantry_vel_y"] = gantry_vel_y
        action["gantry_vel_z"] = gantry_vel_z
        action["winch_line_speed"] = winch_line_speed
        action["finger_angle"] = finger_angle

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Updates the pipeline's feature definition after this processing step."""
        # First, remove the phone-specific features that we consumed.
        for feat in ["enabled", "pos", "rot", "raw_inputs"]:
            features[PipelineFeatureType.ACTION].pop(f"phone.{feat}", None)

        # Second, add the new Stringman-specific features that we created.
        stringman_feats = [
            "gantry_vel_x",
            "gantry_vel_y",
            "gantry_vel_z",
            "winch_line_speed",
            "finger_angle",
        ]
        for feat in stringman_feats:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
