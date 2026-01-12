import serial
import time
import struct

class SimpleSTS3215:
    """
    A lightweight driver for STS3215 servos on clean Half-Duplex UART (No Echo).
    """
    
    # Memory Addresses
    ADDR_ID = 5
    ADDR_BAUD_RATE = 6
    ADDR_MODE = 33
    ADDR_TORQUE_ENABLE = 40
    ADDR_ACC = 41
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_TIME = 44
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56
    ADDR_PRESENT_SPEED = 58
    ADDR_PRESENT_LOAD = 60
    ADDR_PRESENT_VOLTAGE = 62
    ADDR_PRESENT_TEMP = 63
    ADDR_MOVING = 66
    
    def __init__(self, port='/dev/serial0', baudrate=1000000, timeout=0.1):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            rtscts=False,
            dsrdtr=False
        )
        self.header = [0xFF, 0xFF]

    def _calc_checksum(self, data):
        """Standard STS Checksum: Bitwise NOT of sum of parameters."""
        total = sum(data)
        return (~total) & 0xFF

    def _write_packet(self, servo_id, instruction, params):
        """Builds and sends a packet. No Echo handling."""
        # Length = Instruction(1) + Params(N) + Checksum(1)
        length = len(params) + 2
        content = [servo_id, length, instruction] + params
        checksum = self._calc_checksum(content)
        packet = self.header + content + [checksum]
        
        self.ser.reset_input_buffer()
        self.ser.write(bytearray(packet))

    def _read_packet(self, expected_params_len):
        """Reads a response packet with robust header search."""
        # Header(2) + ID(1) + Len(1) + Err(1) + Params(N) + Checksum(1)
        # Total packet length = 6 overhead + params
        total_len = 6 + expected_params_len
        
        # Read enough bytes to cover the packet. 
        # We might need to read slightly more if there's noise, but strict length is safer for now.
        data = self.ser.read(total_len)
        
        if len(data) < total_len:
            raise TimeoutError(f"Expected {total_len} bytes, got {len(data)}")

        # Safety: Check for Header. If data[0] isn't 0xFF, we might have noise.
        # Simple shift logic to find the frame if it's slightly offset.
        while len(data) >= total_len:
            if data[0] == 0xFF and data[1] == 0xFF:
                # Found valid header
                break
            else:
                # Shift buffer by 1 byte and try to read one more to complete the frame
                # This handles the case where a single noise byte appeared before the packet
                next_byte = self.ser.read(1)
                if not next_byte:
                    raise ValueError(f"Invalid Header: {data.hex().upper()}")
                data = data[1:] + next_byte
        
        if data[0] != 0xFF or data[1] != 0xFF:
             raise ValueError(f"Invalid Header after search: {data.hex().upper()}")
             
        # Check error byte (index 4)
        error = data[4]
        if error != 0:
            # You can decide to raise here or just return it. 
            # Printing it is useful for debugging voltage/overload issues.
            print(f"[WARN] Servo Error Byte: {error} (Check Voltage/Overload)")

        return data[5:-1] # Return just the parameters

    def ping(self, servo_id):
        self._write_packet(servo_id, 1, [])
        try:
            # Ping response has 0 params
            self._read_packet(0)
            return True
        except (TimeoutError, ValueError):
            return False

    def set_position(self, servo_id, position, speed=2400, acc=50):
        """
        Moves servo to position (0-4095).
        Speed: Steps/sec (default 2400)
        Acc: Steps/sec^2 (default 50)
        """
        position = max(0, min(4095, position))
        
        # Split integers into Low/High bytes
        pos_L, pos_H = position & 0xFF, (position >> 8) & 0xFF
        spd_L, spd_H = speed & 0xFF, (speed >> 8) & 0xFF
        
        # 1. Set Acceleration (Addr 41)
        self._write_packet(servo_id, 3, [self.ADDR_ACC, acc])
        
        # 2. Set Position & Speed. 
        # Writing to Address 46 (Speed) then 42 (Pos) is standard practice.
        self._write_packet(servo_id, 3, [self.ADDR_GOAL_SPEED, spd_L, spd_H])
        self._write_packet(servo_id, 3, [self.ADDR_GOAL_POSITION, pos_L, pos_H])

    def get_position(self, servo_id):
        # Instruction 2 (Read), Addr 56, Read 2 bytes
        self._write_packet(servo_id, 2, [self.ADDR_PRESENT_POSITION, 2])
        data = self._read_packet(2)
        return (data[1] << 8) | data[0]

    def get_feedback(self, servo_id):
        """Returns Position, Speed, Load, Voltage, Temp, MovingStatus"""
        # Read 11 bytes starting from Position (Addr 56) to Moving (Addr 66)
        # 56-57: Pos, 58-59: Spd, 60-61: Load, 62: Volt, 63: Temp, 64: Reg, 65: Status, 66: Moving
        self._write_packet(servo_id, 2, [self.ADDR_PRESENT_POSITION, 11])
        d = self._read_packet(11)
        
        # Helper for signed 16-bit
        def to_signed(low, high):
            val = (high << 8) | low
            if val > 32767: val -= 65536
            return val

        return {
            "position": (d[1] << 8) | d[0],
            "speed": to_signed(d[2], d[3]),
            "load": to_signed(d[4], d[5]),
            "voltage": d[6] / 10.0,
            "temp": d[7],
            "moving": d[10]
        }
        
    def torque_enable(self, servo_id, enable=True):
        val = 1 if enable else 0
        self._write_packet(servo_id, 3, [self.ADDR_TORQUE_ENABLE, val])

    def set_mode(self, servo_id, mode):
        """
        0: Position Control (Default)
        1: Speed Control (Wheel Mode) -> Requires turning off Torque first!
        2: PWM Control
        """
        # We must disable torque to change mode
        self.torque_enable(servo_id, False)
        self._write_packet(servo_id, 3, [self.ADDR_MODE, mode])
        # Re-enable torque is usually required by user, but let's leave it off for safety

    def set_id(self, current_id, new_id):
        # Unlock EEPROM
        self._write_packet(current_id, 3, [55, 0]) 
        # Write ID (Addr 5)
        self._write_packet(current_id, 3, [self.ADDR_ID, new_id])
        # Lock EEPROM
        self._write_packet(new_id, 3, [55, 1])

# --- Usage Example ---
if __name__ == "__main__":
    sts = SimpleSTS3215(port='/dev/serial0')
    
    print("Pinging ID 1...")
    if sts.ping(1):
        print("Success.")
        sts.torque_enable(1, True)
        
        pos = sts.get_position(1)
        print(f"Current Pos: {pos}")
        
        print("Moving to 2000...")
        sts.set_position(1, 2000)
        
        # Wait for move to complete (simple poll)
        while True:
            feedback = sts.get_feedback(1)
            print(f"Pos: {feedback['position']} | Load: {feedback['load']} | Moving: {feedback['moving']}")
            if feedback['moving'] == 0:
                break
            time.sleep(0.1)
        
        print("Move Complete.")
        sts.torque_enable(1, False)
    else:
        print("Servo not found.")