import serial
import time
import struct
import random
import threading

class SimpleSTS3215:
    """
    A lightweight, thread-safe driver for STS3215 servos on clean Half-Duplex UART.
    
    This driver handles the communication protocol for Feetech STS3215 serial bus servos.
    It is designed for "clean" hardware setups (e.g., dedicated half-duplex adapter or
    properly configured Pi UART) where the transmitted data is NOT echoed back to the RX pin.
    
    Attributes:
        ser (serial.Serial): The underlying pyserial object.
        lock (threading.RLock): Reentrant lock to ensure atomic transactions.
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
    
    def __init__(self, port='/dev/ttyAMA0', baudrate=1000000, timeout=0.5):
        """
        Initializes the serial connection to the servo bus.

        Args:
            port (str): The serial port path (e.g., '/dev/serial0' or 'COM3').
            baudrate (int): Communication speed in bps. Default is 1,000,000.
            timeout (float): Read timeout in seconds. Default 0.5s is conservative for Pi Zero.
        """
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
        self.lock = threading.RLock() # Reentrant lock for nested calls

    def _calc_checksum(self, data):
        """
        Calculates the standard STS protocol checksum.
        
        Formula: ~(Sum of all parameters) & 0xFF

        Args:
            data (list[int]): List of bytes to checksum.

        Returns:
            int: The calculated checksum byte.
        """
        total = sum(data)
        return (~total) & 0xFF

    def _write_packet(self, servo_id, instruction, params):
        """
        Builds and sends a communication packet to the servo.
        
        Includes retry logic for WRITE instructions that expect an ACK.
        
        Args:
            servo_id (int): The target servo ID (0-253) or Broadcast (254).
            instruction (int): The protocol instruction ID.
            params (list[int]): A list of data bytes to send as parameters.
        """
        # Determine if we expect a response (Write instruction to unicast ID)
        expect_ack = (servo_id != 0xFE) and (instruction == 3)
        retries = 3 if expect_ack else 1

        for attempt in range(retries):
            with self.lock:
                # Length = Instruction(1) + Params(N) + Checksum(1)
                length = len(params) + 2
                content = [servo_id, length, instruction] + params
                checksum = self._calc_checksum(content)
                packet = self.header + content + [checksum]
                
                # Clear buffer before sending to ensure we read OUR response
                self.ser.reset_input_buffer()
                self.ser.write(bytearray(packet))
                
                if expect_ack:
                    try:
                        self._read_packet(0)
                        return # Success
                    except (TimeoutError, ValueError):
                        if attempt == retries - 1:
                            raise # Propagate error on last attempt
                        # If failed, loop will retry the write
                        time.sleep(0.01) # Brief pause before retry
                else:
                    # For non-ack packets (like read requests handled by caller), return immediately
                    return

    def _read_packet(self, expected_params_len):
        """
        Reads and validates a response packet from the serial bus.
        
        Includes robust header searching to handle occasional line noise.

        Args:
            expected_params_len (int): The number of data bytes expected in the payload.

        Returns:
            bytes: The payload parameters extracted from the packet.
        """
        # Header(2) + ID(1) + Len(1) + Err(1) + Params(N) + Checksum(1)
        total_len = 6 + expected_params_len
        
        # Read enough bytes to cover the packet. 
        data = self.ser.read(total_len)
        
        if len(data) < total_len:
            raise TimeoutError(f"Expected {total_len} bytes, got {len(data)}: {data.hex().upper()}")

        # Safety: Check for Header. If data[0] isn't 0xFF, we might have noise.
        # We try to slide the window if we have extra bytes, or just fail if strict.
        while len(data) >= total_len:
            if data[0] == 0xFF and data[1] == 0xFF:
                break
            else:
                # Shift buffer by 1 byte and try to read one more
                next_byte = self.ser.read(1)
                if not next_byte:
                    raise ValueError(f"Invalid Header: {data.hex().upper()}")
                data = data[1:] + next_byte
        
        if data[0] != 0xFF or data[1] != 0xFF:
             raise ValueError(f"Invalid Header after search: {data.hex().upper()}")
             
        error = data[4]
        if error != 0:
            error_desc = []
            if error & 1: error_desc.append("Voltage")
            if error & 2: error_desc.append("Angle")
            if error & 4: error_desc.append("Overheat")
            if error & 8: error_desc.append("Range")
            if error & 16: error_desc.append("Checksum")
            if error & 32: error_desc.append("Overload")
            print(f"[WARN] Servo Error: {error} ({', '.join(error_desc)})")

        return data[5:-1] # Return just the parameters

    def ping(self, servo_id):
        """Checks if a servo is present."""
        for _ in range(3):
            try:
                # Manual write packet construction to handle the Read logic purely here
                length = 2
                instruction = 1 # Ping
                content = [servo_id, length, instruction]
                checksum = self._calc_checksum(content)
                packet = self.header + content + [checksum]
                
                with self.lock:
                    self.ser.reset_input_buffer()
                    self.ser.write(bytearray(packet))
                    self._read_packet(0)
                return True
            except (TimeoutError, ValueError):
                time.sleep(0.01)
        return False

    def scan(self, maxid=253):
        """Scans IDs 0-maxid and returns a list of found servos."""
        found = []
        print("Scanning bus for servos...")
        # Reduce timeout temporarily for faster scan
        old_timeout = self.ser.timeout
        self.ser.timeout = 0.05
        
        for i in range(maxid):
            if self.ping(i):
                print(f"Found Servo ID: {i}")
                found.append(i)
        
        self.ser.timeout = old_timeout
        return found

    def change_id(self, current_id, new_id):
        """
        Changes the ID of a connected servo.
        
        WARNING: If multiple servos have 'current_id', they will ALL change.
        Connect one servo at a time to use this safely.
        """
        print(f"Changing ID {current_id} to {new_id}...")
        # Unlock EEPROM
        self._write_packet(current_id, 3, [55, 0]) 
        # Write ID (Addr 5)
        self._write_packet(current_id, 3, [self.ADDR_ID, new_id])
        # Lock EEPROM
        self._write_packet(new_id, 3, [55, 1])
        print("Done.")

    def set_position(self, servo_id, position, speed=2400, acc=50):
        """Moves the servo to a specific target position."""
        position = int(position)
        position = max(0, min(4095, position))
        
        pos_L, pos_H = position & 0xFF, (position >> 8) & 0xFF
        spd_L, spd_H = speed & 0xFF, (speed >> 8) & 0xFF
        
        # We assume _write_packet handles retries for individual packets.
        # However, for set_position, we want the group to succeed.
        # Since _write_packet retries internally, this simple sequence is robust.
        with self.lock:
            self._write_packet(servo_id, 3, [self.ADDR_ACC, acc])
            self._write_packet(servo_id, 3, [self.ADDR_GOAL_SPEED, spd_L, spd_H])
            self._write_packet(servo_id, 3, [self.ADDR_GOAL_POSITION, pos_L, pos_H])

    def set_speed(self, servo_id, speed):
        """
        Sets the target speed for the servo.

        In Position Mode: Limits the maximum speed for the trajectory.
        In Speed Mode (Wheel Mode): Sets the continuous rotation speed and direction.

        Args:
            servo_id (int): The ID of the servo.
            speed (int): Target speed in steps/sec. 
                         - Positive: Clockwise (usually).
                         - Negative: Counter-Clockwise (Bit 15 set).
                         - Range: ~ -3400 to +3400.
        """
        # STS Protocol uses Bit 15 for direction, Bits 0-14 for magnitude
        direction_bit = 0
        if speed < 0:
            speed = abs(speed)
            direction_bit = (1 << 15)
            
        speed = min(speed, 4095) # Clamp magnitude if needed
        
        final_speed_val = speed | direction_bit
        
        spd_L = final_speed_val & 0xFF
        spd_H = (final_speed_val >> 8) & 0xFF
        
        with self.lock:
             self._write_packet(servo_id, 3, [self.ADDR_GOAL_SPEED, spd_L, spd_H])

    def get_position(self, servo_id):
        """Reads the current absolute position of the servo shaft."""
        for _ in range(3):
            try:
                # Instruction 2 (Read)
                length = 4 
                instruction = 2
                params = [self.ADDR_PRESENT_POSITION, 2]
                content = [servo_id, length, instruction] + params
                checksum = self._calc_checksum(content)
                packet = self.header + content + [checksum]
                
                with self.lock:
                    self.ser.reset_input_buffer()
                    self.ser.write(bytearray(packet))
                    data = self._read_packet(2)
                return (data[1] << 8) | data[0]
            except (TimeoutError, ValueError):
                time.sleep(0.01)
        raise TimeoutError(f"Failed to read position from ID {servo_id}")

    def get_feedback(self, servo_id):
        """Retrieves a comprehensive status snapshot of the servo."""
        for _ in range(3):
            try:
                length = 4
                instruction = 2
                params = [self.ADDR_PRESENT_POSITION, 11]
                content = [servo_id, length, instruction] + params
                checksum = self._calc_checksum(content)
                packet = self.header + content + [checksum]
                
                with self.lock:
                    self.ser.reset_input_buffer()
                    self.ser.write(bytearray(packet))
                    d = self._read_packet(11)
                
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
            except (TimeoutError, ValueError):
                time.sleep(0.01)
        raise TimeoutError(f"Failed to read feedback from ID {servo_id}")
        
    def torque_enable(self, servo_id, enable=True):
        """Enables or disables the motor torque."""
        val = 1 if enable else 0
        self._write_packet(servo_id, 3, [self.ADDR_TORQUE_ENABLE, val])

    def set_mode(self, servo_id, mode):
        """Sets the operational mode of the servo."""
        self.torque_enable(servo_id, False)
        self._write_packet(servo_id, 3, [self.ADDR_MODE, mode])

if __name__ == "__main__":
    sts = SimpleSTS3215(port='/dev/serial0', timeout=0.5) 

    # --- CLI Mode: Set ID ---
    if len(sys.argv) == 3 and sys.argv[1] == "setid":
        try:
            target_new_id = int(sys.argv[2])
            if not (0 <= target_new_id <= 253):
                raise ValueError("ID must be 0-253")
        except ValueError as e:
            print(f"Invalid ID: {e}")
            sys.exit(1)

        print(f"Attempting to set single connected servo to ID {target_new_id}...")
        
        # Perform a full scan (0-253) to ensure we don't accidentally broadcast to multiple motors
        # If we see ANY more than 1 motor, we abort to prevent ID collisions.
        servos = sts.scan() 
        
        if len(servos) == 0:
            print("Error: No servos found. Check power and connections.")
        elif len(servos) > 1:
            print(f"Error: Found {len(servos)} servos ({servos}).")
            print("SAFETY ABORT: Changing ID with multiple servos connected will change ALL of them to the same ID.")
            print("Please connect ONLY the specific servo you want to configure.")
        else:
            current_id = servos[0]
            print(f"Found exactly one servo at ID {current_id}.")
            
            if current_id == target_new_id:
                print(f"Servo is already set to ID {target_new_id}. No action needed.")
            else:
                sts.change_id(current_id, target_new_id)

    # --- Default Mode: Scan and Test IDs 1-8 ---
    else:
        servos = sts.scan(8)
        
        if not servos:
            print("No servos found. Check power/wiring.")
            print("Usage: python simple_sts3215.py")
            print("       python simple_sts3215.py setid <new_id>")
            exit()
            
        print(f"Active Servos: {servos}")

        for target_id in servos:
            print(f"Testing Servo ID {target_id}...")
            sts.torque_enable(target_id, True)
            
            pos = sts.get_position(target_id)
            print(f"Current Pos: {pos}")
            
            target_pos = random.randint(500, 3500)
            print(f"Moving to {target_pos}...")
            sts.set_position(target_id, target_pos)
            
            for _ in range(20): 
                try:
                    feedback = sts.get_feedback(target_id)
                    print(f"Pos: {feedback['position']} | Load: {feedback['load']} | Moving: {feedback['moving']}")
                    if abs(feedback['position'] - target_pos) < 15:
                        print("Target Reached!")
                        break
                except Exception as e:
                    print(f"Feedback read error: {e}")
                time.sleep(0.1)
            
            sts.torque_enable(target_id, False)