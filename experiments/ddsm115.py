import RPi.GPIO as GPIO
import serial
import struct
import time

class DDSM115:
    """
    Controller for Waveshare DDSM115 Direct Drive Servo Motor.
    Handles RS485 half-duplex switching and the CRC8/MAXIM protocol.
    """
    
    # Motor Mode Values
    MODE_CURRENT = 0x01
    MODE_VELOCITY = 0x02
    MODE_POSITION = 0x03

    def __init__(self, port="/dev/ttyAMA0", baudrate=115200, en_pin=4, motor_id=1):
        self.motor_id = motor_id
        self.en_pin = en_pin
        
        # Initialize GPIO for RS485 flow control
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.en_pin, GPIO.OUT)
        GPIO.output(self.en_pin, GPIO.LOW) # Default to listening

        self.ser = serial.Serial(port, baudrate, timeout=0.1)

    def _crc8_maxim(self, data):
        """
        Calculates CRC-8/MAXIM checksum.
        Polynomial: x8 + x5 + x4 + 1 (0x31), Initial: 0x00, RefIn: True, RefOut: True.
        """
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x01:
                    crc = (crc >> 1) ^ 0x8C # 0x8C is the reflection of 0x31
                else:
                    crc >>= 1
        return crc

    def _send_command(self, packet):
        """
        Manages the RS485 Transmit Enable pin and sends the 10-byte packet.
        """
        if len(packet) != 10:
            raise ValueError("Packet must be exactly 10 bytes")

        GPIO.output(self.en_pin, GPIO.HIGH)
        self.ser.write(packet)
        # Flush ensures the write buffer is pushed to the OS level
        self.ser.flush()
        
        # Small delay to allow the physical UART bits to leave the wire
        # before we pull the Enable pin low to listen for the reply.
        time.sleep(0.001) 
        GPIO.output(self.en_pin, GPIO.LOW)

    def _receive_feedback(self):
        """
        Reads and parses the 10-byte response from the motor.
        """
        data = self.ser.read(10)
        if len(data) < 10:
            return None
        
        if self._crc8_maxim(data[:-1]) != data[9]:
            return None

        # Determine if this is a Protocol 2 response (mode byte 0x74)
        is_diag = data[1] == 0x74

        return {
            "id": data[0],
            "mode": data[1],
            "current": struct.unpack('>h', data[2:4])[0] * (8.0 / 32767.0), # Amps
            "velocity": struct.unpack('>h', data[4:6])[0],                # RPM
            # Protocol 1 uses 16-bit pos, Protocol 2 uses 8-bit pos in byte 7
            "position_deg": (struct.unpack('>H', data[6:8])[0] * (360.0 / 32767.0)) if not is_diag else (data[7] * (360.0 / 255.0)),
            "temp_c": data[6] if is_diag else None,
            "error_code": data[8]
        }

    def set_mode(self, mode):
        """
        Switches motor between Current, Velocity, or Position loops.
        Note: Velocity must be < 10rpm to switch to Position mode.
        """
        packet = bytearray([self.motor_id, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, mode])
        self._send_command(packet)

    def drive(self, value, acceleration=0, brake=0x00):
        """
        Sends drive command. Meaning of 'value' depends on current mode:
        - Velocity Mode: rpm (-330 to 330)
        - Current Mode: scaled value (-32767 to 32767)
        - Position Mode: scaled degrees (0 to 32767)
        """
        val_bytes = struct.pack('>H', int(value))
        packet = bytearray([self.motor_id, 0x64, val_bytes[0], val_bytes[1], 0x00, acceleration, 0x00, brake, 0x00, 0x00])
        packet[9] = self._crc8_maxim(packet[:-1])
        
        self._send_command(packet)
        return self._receive_feedback()

    def brake(self):
        """Emergency stop command for velocity loop."""
        return self.drive(0, brake=0xFF)

    def get_feedback(self):
        """Queries motor for temperature and detailed status (Protocol 2)."""
        packet = bytearray([self.motor_id, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        packet[9] = self._crc8_maxim(packet[:-1])
        
        self._send_command(packet)
        return self._receive_feedback()

    def set_id(self, new_id):
        """Sets the motor ID. Only one motor should be connected."""
        packet = bytearray([0xAA, 0x55, 0x53, new_id, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        packet[9] = self._crc8_maxim(packet[:-1])
        for _ in range(5):
            self._send_command(packet)
            time.sleep(0.1)

if __name__ == "__main__":
    motor = DDSM115(en_pin=4)
    
    # 1. Get initial Diagnostics
    print("--- Diagnostic Info ---")
    diag = motor.get_feedback()
    if diag:
        print(f"Motor ID: {diag['id']}")
        print(f"Temperature: {diag['temp_c']}°C")
        print(f"Error Code: {hex(diag['error_code'])}")
        print(f"Current Position: {diag['position_deg']:.2f}°")
    
    # 2. Setup Position Mode
    # Ensure motor is slow enough to switch to position mode safely
    motor.set_mode(DDSM115.MODE_VELOCITY)
    motor.drive(0)
    time.sleep(0.5)
    
    print("\nSwitching to Position Mode...")
    motor.set_mode(DDSM115.MODE_POSITION)
    time.sleep(0.1)
    
    # 3. Movement sequence
    # Note: Position 0-32767 maps to 0-360 degrees.
    # We'll work relative to the current position.
    start_pos_deg = diag['position_deg'] if diag else 0
    
    def move_to_angle(angle_deg):
        # Wrap angle to 0-360
        target_angle = angle_deg % 360
        # Convert degrees to 0-32767 range
        raw_val = int((target_angle / 360.0) * 32767)
        print(f"Moving to {target_angle:.2f}° (Raw: {raw_val})")
        return motor.drive(raw_val)

    # Move +20 degrees
    status = move_to_angle(start_pos_deg + 100)
    time.sleep(1.5) # Give time for physical movement
    
    # Move -20 degrees (back to start)
    status = move_to_angle(start_pos_deg)
    time.sleep(1.5)

    print("\nFinal Status:", motor.get_feedback())