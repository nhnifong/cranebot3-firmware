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
    
    def __init__(self, port='/dev/serial0', baudrate=1000000, timeout=0.5):
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
        
        This method consumes the acknowledgment response immediately 
        to keep the serial buffer clean.

        Args:
            servo_id (int): The target servo ID (0-253) or Broadcast (254).
            instruction (int): The protocol instruction ID (e.g., 3 for Write, 2 for Read).
            params (list[int]): A list of data bytes to send as parameters.
        """
        with self.lock:
            # Length = Instruction(1) + Params(N) + Checksum(1)
            length = len(params) + 2
            content = [servo_id, length, instruction] + params
            checksum = self._calc_checksum(content)
            packet = self.header + content + [checksum]
            
            # Clear buffer before sending to ensure we read OUR response, not old noise
            self.ser.reset_input_buffer()
            self.ser.write(bytearray(packet))
            
            # If this is NOT a broadcast (ID 254/0xFE), the servo WILL reply.
            # We must read that reply now, or it will clog the buffer for later reads.
            if servo_id != 0xFE:
                # Write instructions (Inst=3) return a status packet with 0 params.
                # Read instructions (Inst=2) are handled by the calling function.
                if instruction == 3: 
                    self._read_packet(0)

    def _read_packet(self, expected_params_len):
        """
        Reads and validates a response packet from the serial bus.
        
        Includes robust header searching to handle occasional line noise.

        Args:
            expected_params_len (int): The number of data bytes expected in the payload.
                                       (e.g., Ping=0, ReadPosition=2).

        Returns:
            bytes: The payload parameters extracted from the packet.

        Raises:
            TimeoutError: If insufficient bytes are received.
            ValueError: If the header is invalid or checksum fails (checksum check not implemented for speed).
        """
        # Header(2) + ID(1) + Len(1) + Err(1) + Params(N) + Checksum(1)
        # Total packet length = 6 overhead + params
        total_len = 6 + expected_params_len
        
        # Read enough bytes to cover the packet. 
        data = self.ser.read(total_len)
        
        if len(data) < total_len:
            # Raise exception to see trace, but include what we did get
            raise TimeoutError(f"Expected {total_len} bytes, got {len(data)}: {data.hex().upper()}")

        # Safety: Check for Header. If data[0] isn't 0xFF, we might have noise.
        while len(data) >= total_len:
            if data[0] == 0xFF and data[1] == 0xFF:
                # Found valid header
                break
            else:
                # Shift buffer by 1 byte and try to read one more
                next_byte = self.ser.read(1)
                if not next_byte:
                    raise ValueError(f"Invalid Header: {data.hex().upper()}")
                data = data[1:] + next_byte
        
        if data[0] != 0xFF or data[1] != 0xFF:
             raise ValueError(f"Invalid Header after search: {data.hex().upper()}")
             
        # Check error byte (index 4)
        error = data[4]
        if error != 0:
            # Bit 0: Voltage, Bit 1: Angle, Bit 2: Overheat, Bit 3: Range, Bit 4: Checksum, Bit 5: Overload
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
        """
        Checks if a servo is present and communicating.

        Args:
            servo_id (int): The ID of the servo to ping (0-253).

        Returns:
            bool: True if the servo responds, False otherwise.
        """
        # Ping uses Instruction 1. The response has 0 params.
        length = 2
        instruction = 1 # Ping
        content = [servo_id, length, instruction]
        checksum = self._calc_checksum(content)
        packet = self.header + content + [checksum]
        
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(bytearray(packet))
            
            try:
                self._read_packet(0)
                return True
            except (TimeoutError, ValueError):
                return False

    def set_position(self, servo_id, position, speed=2400, acc=50):
        """
        Moves the servo to a specific target position.
        
        This method atomically sets Acceleration, Speed, and Position in a single
        locked transaction to ensure smooth motion.

        Args:
            servo_id (int): The ID of the servo.
            position (int): Target position in steps. 
                            Range: 0-4095 (0 to 360 degrees, ~0.088° per step).
            speed (int, optional): Movement speed in steps/second. 
                                   Range: ~0-3000+. Default 2400.
            acc (int, optional): Acceleration in steps/second^2. 
                                 Range: 0-254. Default 50.
        """
        position = max(0, min(4095, position))
        
        pos_L, pos_H = position & 0xFF, (position >> 8) & 0xFF
        spd_L, spd_H = speed & 0xFF, (speed >> 8) & 0xFF
        
        # Lock the entire sequence so Acceleration, Speed, and Position are set atomically
        with self.lock:
            # We send 3 commands. _write_packet will automatically read the ACK for each.
            self._write_packet(servo_id, 3, [self.ADDR_ACC, acc])
            self._write_packet(servo_id, 3, [self.ADDR_GOAL_SPEED, spd_L, spd_H])
            self._write_packet(servo_id, 3, [self.ADDR_GOAL_POSITION, pos_L, pos_H])

    def get_position(self, servo_id):
        """
        Reads the current absolute position of the servo shaft.

        Args:
            servo_id (int): The ID of the servo.

        Returns:
            int: Current position in steps (0-4095).
        """
        # Instruction 2 (Read), Addr 56, Read 2 bytes
        length = 4 # Inst(1) + Addr(1) + ReadLen(1) + Chk(1)
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

    def get_feedback(self, servo_id):
        """
        Retrieves a comprehensive status snapshot of the servo.

        Args:
            servo_id (int): The ID of the servo.

        Returns:
            dict: A dictionary containing:
                - 'position' (int): Current position [0-4095 steps]
                - 'speed' (int): Current speed [steps/s, signed]
                - 'load' (int): Current load/current [0-1000, approx 0-100% torque]
                - 'voltage' (float): Input voltage [Volts]
                - 'temp' (int): Internal temperature [°C]
                - 'moving' (int): 1 if moving, 0 if stopped
        """
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
        
    def torque_enable(self, servo_id, enable=True):
        """
        Enables or disables the motor torque.

        Args:
            servo_id (int): The ID of the servo.
            enable (bool): True to enable torque (stiff), False to release (free-spinning).
        """
        val = 1 if enable else 0
        # _write_packet handles the locking
        self._write_packet(servo_id, 3, [self.ADDR_TORQUE_ENABLE, val])

    def set_mode(self, servo_id, mode):
        """
        Sets the operational mode of the servo.
        
        Note: Torque is automatically disabled before changing modes as per protocol requirements.

        Args:
            servo_id (int): The ID of the servo.
            mode (int): 
                0: Position Control Mode (Standard Servo behavior, 0-360°)
                1: Speed Control Mode (Continuous Rotation / Wheel Mode)
                2: PWM Control Mode
                3: Step Servo Mode
        """
        # _write_packet handles the locking
        self.torque_enable(servo_id, False)
        self._write_packet(servo_id, 3, [self.ADDR_MODE, mode])

# --- Usage Example ---
if __name__ == "__main__":
    sts = SimpleSTS3215(port='/dev/serial0', timeout=0.5) 
    
    print("Pinging ID 1...")
    if sts.ping(1):
        print("Success.")
        sts.torque_enable(1, True)
        
        # Check initial position
        pos = sts.get_position(1)
        print(f"Current Pos: {pos}")
        
        # Move to random position
        target_pos = random.randint(500, 3500)
        print(f"Moving to {target_pos}...")
        sts.set_position(1, target_pos)
        
        # Wait for move
        for _ in range(20): # Timeout after ~2 seconds
            feedback = sts.get_feedback(1)
            print(f"Pos: {feedback['position']} | Load: {feedback['load']} | Moving: {feedback['moving']}")
            
            # Simple "reached target" check (allow small error margin)
            if abs(feedback['position'] - target_pos) < 10:
                print("Target Reached!")
                break
            time.sleep(0.1)
        
        sts.torque_enable(1, False)
    else:
        print("Servo not found.")