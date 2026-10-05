"""I2C addresses of the devices on the Stringman Gripper Hat.

Kept free of hardware library imports so the dispatcher can identify a gripper without
loading the gripper server.
"""

VL53L1X_ADDR = 0x29 # rangefinder
ADS1015_ADDR = 0x48 # finger pad pressure ADC
MPU6050_ADDR = 0x68 # IMU on the original hats
# IMU on newer hats. 0x6A with SDO low, 0x6B with it high.
LSM6DS3TRC_ADDRS = (0x6A, 0x6B)
IMU_ADDRS = {MPU6050_ADDR, *LSM6DS3TRC_ADDRS}

IMU_MPU6050 = 'MPU6050'
IMU_LSM6DS3TRC = 'LSM6DS3TR-C'


def find_imu(addrs):
    """(IMU type, address) for the IMU among the scanned addresses, or None if there is none."""
    for addr in LSM6DS3TRC_ADDRS:
        if addr in addrs:
            return IMU_LSM6DS3TRC, addr
    if MPU6050_ADDR in addrs:
        return IMU_MPU6050, MPU6050_ADDR
    return None

