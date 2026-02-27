import can
import time
import math
import numpy as np
from enum import IntEnum
from struct import unpack, pack

# you must run the following after booting the pi
# sudo ip link set can0 up type can bitrate 1000000
# sudo ifconfig can0 txqueuelen 65536

# Defaulting to 1Mbps as initialized in your previous terminal session.
BITRATE = 1000000 

class DM_Motor_Type(IntEnum):
    DM4310 = 0
    DM4310_48V = 1
    DM4340 = 2
    DM4340_48V = 3
    DM6006 = 4
    DM8006 = 5
    DM8009 = 6
    DM10010L = 7
    DM10010 = 8
    DMH3510 = 9
    DMH6215 = 10
    DMG6220 = 11

class Control_Type(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    Torque_Pos = 4

class DM_variable(IntEnum):
    UV_Value = 0       
    OV_Value = 29      
    CTRL_MODE = 10
    V_BUS = 81         
    PMAX = 21
    VMAX = 22
    TMAX = 23

class Motor_Error(IntEnum):
    NONE = 0
    OVER_VOLTAGE = 1
    UNDER_VOLTAGE = 2
    OVER_CURRENT = 3
    OVER_TEMP = 4
    MAGNETIC_ERR = 5
    CONNECTION_ERR = 6

def uint_to_float(x, x_min, x_max, bits):
    return np.float32((float(x) / ((1 << bits) - 1)) * (x_max - x_min) + x_min)

class Motor:
    def __init__(self, MotorType, SlaveID, MasterID):
        self.state_q = 0.0
        self.state_dq = 0.0
        self.state_torque = 0.0
        self.last_error = Motor_Error.NONE
        self.SlaveID = SlaveID
        self.MasterID = MasterID
        self.MotorType = MotorType

    def recv_data(self, q, dq, torque, error_code):
        self.state_q, self.state_dq, self.state_torque = q, dq, torque
        try:
            self.last_error = Motor_Error(error_code & 0x0F)
        except ValueError:
            pass

class MotorControl:
    # H6215 limits: Pos +/- 12.5 rad, Vel +/- 45 rad/s, Torque +/- 10 Nm
    # Note: Multi-turn position is tracked internally by the motor and reported in the 16-bit field
    # but the scaling is based on the PMAX parameter.
    Limit_Param = [[12.5, 30, 10], [12.5, 50, 10], [12.5, 8, 28], [12.5, 10, 28],
                   [12.5, 45, 20], [12.5, 45, 40], [12.5, 45, 54], [12.5, 25, 200], [12.5, 20, 200],
                   [12.5, 280, 1], [12.5, 45, 10], [12.5, 45, 10]]

    def __init__(self, channel='can0'):
        self.bus = can.interface.Bus(channel=channel, interface='socketcan', bitrate=BITRATE)
        self.motors_map = {}

    def _send_can_frame(self, arbitration_id, data):
        self.bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False))

    def addMotor(self, motor):
        self.motors_map[motor.SlaveID] = motor

    def enable(self, motor):
        # 0xFB (Clear), 0xFE (Zero), 0xFC (Enable)
        # We clear faults multiple times to ensure the OVER_VOLTAGE status doesn't block the enable bit.
        for _ in range(2):
            self._send_can_frame(motor.SlaveID, [0xFF]*7 + [0xFB])
            time.sleep(0.02)
        self._send_can_frame(motor.SlaveID, [0xFF]*7 + [0xFE])
        time.sleep(0.02)
        self._send_can_frame(motor.SlaveID, [0xFF]*7 + [0xFC])
        time.sleep(0.1)
        self.receive_and_process()

    def disable(self, motor):
        self._send_can_frame(motor.SlaveID, [0xFF]*7 + [0xFD])
        time.sleep(0.1)

    def write_parameter(self, motor, rid, value):
        data = [motor.SlaveID & 0xFF, (motor.SlaveID >> 8) & 0xFF, 0x55, int(rid)] + list(pack('<f', float(value)))
        self._send_can_frame(0x7FF, data)
        time.sleep(0.1)

    def control_Velocity(self, motor, vel_target):
        """
        Velocity Control Mode (ID 0x200 offset).
        vel_target: float in rad/s
        The 0x200 offset expects a float32 for the target velocity in the first 4 bytes.
        """
        data = list(pack('<f', float(vel_target))) + [0, 0, 0, 0]
        self._send_can_frame(0x200 + motor.SlaveID, data)
        self.receive_and_process()

    def receive_and_process(self):
        while True:
            msg = self.bus.recv(timeout=0.001)
            if not msg: break
            # Damiao motors often reply on ID 0x00 or SlaveID. We map to the known motor.
            target_motor = self.motors_map.get(msg.arbitration_id) or next(iter(self.motors_map.values()))
            if len(msg.data) >= 6:
                limits = self.Limit_Param[target_motor.MotorType]
                # Feedback: Byte 0 (Status), Bytes 1-2 (Pos), Bytes 3-4 (Vel), Bytes 5-6 (Torque)
                target_motor.recv_data(
                    uint_to_float(np.uint16((np.uint16(msg.data[1]) << 8) | msg.data[2]), -limits[0], limits[0], 16),
                    uint_to_float(np.uint16((np.uint16(msg.data[3]) << 4) | (msg.data[4] >> 4)), -limits[1], limits[1], 12),
                    uint_to_float(np.uint16(((msg.data[4] & 0xf) << 8) | msg.data[5]), -limits[2], limits[2], 12),
                    msg.data[0]
                )

if __name__ == "__main__":
    my_motor = Motor(DM_Motor_Type.DMH6215, SlaveID=0x01, MasterID=0x00)
    controller = MotorControl()
    controller.addMotor(my_motor)

    print("Enabling motor and switching to VELOCITY mode...")
    controller.enable(my_motor)
    
    # Mode 3 is Velocity Control
    controller.write_parameter(my_motor, DM_variable.CTRL_MODE, 3.0)
    time.sleep(0.5)
    
    print("Starting Velocity Ramping Demo...")
    try:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            # Target velocity oscillating between -10 and +10 rad/s
            target_vel = math.sin(elapsed) * 10.0
            
            controller.control_Velocity(my_motor, target_vel)
            
            status = f" | ERR: {my_motor.last_error.name}" if my_motor.last_error != Motor_Error.NONE else ""
            print(f"Target: {target_vel:+.2f} rad/s | Actual Vel: {my_motor.state_dq:+.3f} | Pos: {my_motor.state_q:+.3f} | Torque: {my_motor.state_torque:+.3f}{status}", end='\r')
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping...")
        controller.control_Velocity(my_motor, 0.0)
        controller.disable(my_motor)
        controller.bus.shutdown()