# runs a self test on an assembled gripper and prints a report on which systems are working correctly.
import time
from inventorhatmini import InventorHATMini, SERVO_1, SERVO_2, ADC
from ioexpander import IN_PU
import board
import busio
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_vl53l1x import VL53L1X

# gpio pin of pressure sensing resistor
PRESSURE_PIN = 0
# gpio pin of limit switch. 0 is pressed
LIMIT_SWITCH_PIN = 1

def main():
    try:
        hat = InventorHATMini(init_leds=False)
        hand_servo = hat.servos[SERVO_2]
        winch_servo = hat.servos[SERVO_1]
        hat.gpio_pin_mode(PRESSURE_PIN, ADC)
        hat.gpio_pin_mode(LIMIT_SWITCH_PIN, IN_PU)
    except:
        print('Failed to initialize Inventor Hat Mini. I2C bus may be noisy, have a bad item connected to it. check continuity of i2c bus line and try removing one i2c item at a time')
        raise


    input("Press Enter to move gripper fingers...")
    # move to neutral position
    hand_servo.value(0)
    # check finger voltage
    voltage = hat.gpio_pin_value(PRESSURE_PIN)
    time.sleep(1.5)
    voltage = hat.gpio_pin_value(PRESSURE_PIN)
    assert voltage==0, f"finger voltage did not measure zero with fingers not touching. (was {voltage}v)"

    try:
        hand_servo.value(90)
        time.sleep(1.5)
        voltage = hat.gpio_pin_value(PRESSURE_PIN)
        assert voltage>0.5, f"finger voltage did not rise above 0.5v after moving to max closed position. Either the fingers did not touch and you need to disassemble and adjust them closed by one tooth or the pressure sense resistor is not connectd."
        print('Finger function and pressure sense are normal')
    finally:
        hand_servo.value(0)
        time.sleep(1)


    assert hat.gpio_pin_value(LIMIT_SWITCH_PIN)==1, "Limit switch pin may be disconnected or wired to the wrong switch terminals. circuit should be closed when switch is depressed."
    print('Please click the limit switch on the top of the gripper...')
    count = 100
    while count>0 and hat.gpio_pin_value(LIMIT_SWITCH_PIN)==1:
        count-=1
        time.sleep(0.05)
    assert count!=0, "Timed out waiting for limit switch click. Switch may not be connected properly."
    print('Limit switch OK')

    winch_servo.value(0)
    start_revs = hat.encoders[0].revolutions()

    input("Press Enter to move winch motor...")
    try:
        winch_servo.value(10)
        time.sleep(1)
        revs = hat.encoders[0].revolutions()
        assert revs > start_revs, "Encoder did not show any winch spool motion. If you heard it move, you may need up update the IHM firmware."
        winch_servo.value(-10)
        time.sleep(1)
    finally:
        winch_servo.value(0)
    print('Winch and encoder ok')

    # confim readings from IMU
    i2c = busio.I2C(board.SCL, board.SDA)
    imu = BNO08X_I2C(i2c, address=0x4b)
    imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)
    time.sleep(0.5)

    start_quat = imu.quaternion
    assert sum(start_quat)!=0, "IMU readings appeard to be zero. it may be dead or there may be a continuity problem in the I2C bus wire"
    print('IMU readings appear normal')

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

    print('All sensors working normally. Camera not checked.')

if __name__ == "__main__":
    main()