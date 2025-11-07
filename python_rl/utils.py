import serial
import json
import time

class ESP32SerialHelper:
    def __init__(self, port='COM12', baudrate=921600, timeout=1.0):
        """
        Initialize serial connection to ESP32
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for ESP32 to boot
        self.ser.reset_input_buffer()

    def send_action(self, action: float):
        """
        Send an action as text via Serial, e.g., '1.23\n'
        """
        self.ser.write(f"{action}\n".encode())

    def read_json_data(self):
        """
        Read one line of JSON from ESP32 and return as dict if valid
        """
        try:
            raw = self.ser.readline()  # bytes
            if not raw:
                return None

            # decode and ignore invalid bytes
            line = raw.decode('utf-8', errors='ignore').strip()
            # only parse if it's a proper JSON object
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e} -> {line}")
        return None

    def close(self):
        self.ser.close()
