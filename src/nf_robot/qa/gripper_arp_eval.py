# runs a self test on an assembled arpeggio gripper and prints a report on which systems are working correctly.

import time
import socket
import subprocess
import board
import busio
import json
# import adafruit_bno08x
# from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_vl53l1x import VL53L1X # rangefinder
from adafruit_ads1x15 import ADS1015, AnalogIn, ads1x15 # analog2digital converter for pressure

from nf_robot.robot.simple_st3215 import SimpleSTS3215

FINGER_MOTOR_ID = 1
WRIST_MOTOR_ID = 2
STEPS_PER_REV = 4096
GEAR_RATIO = 10/45 # a finger lever makes this many revolutions per revolution of the drive gear
FINGER_TRAVEL_DEG = 59 # actually 60 but need small margin of space at wide open. 
FINGER_TRAVEL_STEPS = FINGER_TRAVEL_DEG / 360 / GEAR_RATIO * STEPS_PER_REV

def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    sts = SimpleSTS3215(port='/dev/serial0', timeout=0.5)

    def wiggle_wrist():
        sts.torque_enable(WRIST_MOTOR_ID, True)
        pos = sts.get_position(WRIST_MOTOR_ID)
        sts.set_position(WRIST_MOTOR_ID, pos+200)
        time.sleep(0.5)
        sts.set_position(WRIST_MOTOR_ID, pos)
        time.sleep(0.5)
        sts.torque_enable(WRIST_MOTOR_ID, False)

    input("Plug only the wrist motor into the board and press Enter...")
    # set id of this motor to 2. let the finger motor id remain 1, the factory setting.

    # Perform a full scan (0-253) to ensure we don't accidentally broadcast to multiple motors
    # If we see ANY more than 1 motor, we abort to prevent ID collisions.
    print("Scanning for connected motors...")
    servos = sts.scan(8)

    if len(servos) == 2 and (1 in servos and 2 in servos):
        # spin the wrist motor a small amount
        input(f"Found exactly two servos with IDs 1 and 2. Press enter to move servo 2 and note which it is")
        wiggle_wrist()
        val = input("Did the wrist move? y/n")
        if val == 'y':
            print('servo IDs correct.')
        else:
            input('Servo IDs are backwards will be swapped. press enter to confirm.')
            sts.change_id(WRIST_MOTOR_ID, 3)
            sts.change_id(FINGER_MOTOR_ID, WRIST_MOTOR_ID)
            sts.change_id(3, FINGER_MOTOR_ID)
    elif len(servos) == 0:
        print("Error: No servos found. Check connections.")
        quit()
    elif len(servos) > 1:
        print(f"Error: Found {len(servos)} servos ({servos}).")
        print("Changing ID with multiple servos connected will change ALL of them to the same ID.")
        print("If both have the same id, unplug one and rerun this script")
        quit()
    elif len(servos)==1:
        current_id = servos[0]
        input(f"Found exactly one servo with ID {current_id}. Press enter to move it and note which it is")
        wiggle_wrist()
        val = input("Did the wrist move? y/n")
        if val == 'y':
            if current_id == WRIST_MOTOR_ID:
                print(f"Servo is wrist and already set correctly to {WRIST_MOTOR_ID}.")
            else:
                sts.change_id(current_id, WRIST_MOTOR_ID)
        else:
            if current_id == FINGER_MOTOR_ID:
                print(f"Servo is finger and already set correctly to {FINGER_MOTOR_ID}.")
            else:
                sts.change_id(current_id, WRIST_MOTOR_ID)

    time.sleep(0.1)
    assert sts.ping(FINGER_MOTOR_ID), "Finger motor did not respond to ping"
    assert sts.ping(WRIST_MOTOR_ID), "Wrist motor did not respond to ping"

    ads = ADS1015(i2c)
    pressure_sensor = AnalogIn(ads, ads1x15.Pin.A0)

    input("Press Enter to move gripper fingers...")
    # Find the position at which the gripper fingers touch.
    print(f"Calibrating finger servo...")
    sts.torque_enable(FINGER_MOTOR_ID, True)

    pos = sts.get_position(FINGER_MOTOR_ID)
    # open a few degrees in case fingers were already touching.
    rel = 100
    sts.set_position(FINGER_MOTOR_ID, pos + rel)
    time.sleep(0.5)
    data = sts.get_feedback(FINGER_MOTOR_ID)

    # confirm no pressure on finger pad
    v = pressure_sensor.voltage
    assert v > 3, "Voltage too low on finger pad. Is pressure sensor connected?"

    # slowly close until the fingerpad voltage drops below 2V
    start = time.time()
    load = 0
    while v > 3.0 and time.time() < start+10:
        sts.set_position(FINGER_MOTOR_ID, pos + rel)
        rel -= 10
        time.sleep(0.05)
        v = pressure_sensor.voltage
        data = sts.get_feedback(FINGER_MOTOR_ID)
        load = data["load"]
        print(f'voltage={v}, load={load}')
        if load < 1000: # ignore load values over 1000, they're not real
            assert load<450, "motor load too high while no finger pressure detected"
    sts.set_speed(FINGER_MOTOR_ID, 0)

    finger_closed_pos = sts.get_position(FINGER_MOTOR_ID)
    print(f"Motor encoder position at finger touch = {finger_closed_pos}")
    finger_open_pos = finger_closed_pos + FINGER_TRAVEL_STEPS
    with open('arp_gripper_state.json', 'w') as f:
        json.dump({
            'finger_closed_pos': finger_closed_pos,
            'finger_open_pos': finger_open_pos,
        }, f)

    # move back to a neutral position
    sts.set_position(FINGER_MOTOR_ID, int(finger_closed_pos+FINGER_TRAVEL_STEPS*0.3))
    time.sleep(0.75)
    sts.torque_enable(FINGER_MOTOR_ID, False)

    # confim readings from IMU
    # i2c = busio.I2C(board.SCL, board.SDA)
    # imu = BNO08X_I2C(i2c, address=0x4b)
    # imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)
    # time.sleep(0.5)

    # start_quat = imu.quaternion
    # assert sum(start_quat)!=0, "IMU readings appeard to be zero. it may be dead or there may be a continuity problem in the I2C bus wire"
    # print('IMU readings appear normal')

    # confirm readings from Rangefinder.
    rangefinder = VL53L1X(i2c)
    rangefinder.distance_mode = 2 # LONG. results returned in centimeters.
    rangefinder.start_ranging()
    print('Please put your hand in front of the rangefinger...')
    count = 100
    # note that rangefinder distance is none when exceeding it's range
    while count>0 and rangefinder.distance is None or abs(rangefinder.distance) < 1:
        count-=1
        time.sleep(0.05)
    assert count!=0, "Timed out waiting for change in observed distance from rangefinder. Rangefinger may not be functioning."
    print('Rangefinder readings normal')

    print('All sensors working normally. Starting Camera...')

    stream_command = """
    /usr/bin/rpicam-vid -t 0 -n \
      --width=1920 --height=1080 \
      -o tcp://0.0.0.0:8888?listen=1 \
      --codec libav \
      --libav-format mpegts \
      --autofocus-mode continuous \
      --bitrate 2000kbps
    """.split()

    # get my ip address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    print('Please run the following on your host machine and confirm good video, then close the video window.')
    print(f'ffplay -fast -fflags nobuffer -flags low_delay "tcp://{addr}:8888"')

    subprocess.run(stream_command)


if __name__ == "__main__":
    main()