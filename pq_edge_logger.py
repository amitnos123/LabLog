import csv
import time
import datetime
import os
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException, ConnectionException

class PQube3Logger:
    def __init__(self, host, port=502, interval=0.5):
        self.client = ModbusTcpClient(host, port=port, timeout=3)  # Set timeout
        self.interval = interval
        self.start_address = 7000  # The Modbus start address
        self.registers = {
            'L1-E Voltage': (self.start_address + 0, self.start_address + 1),
            'Frequency': (self.start_address + 26, self.start_address + 27),
            'Ext_Probe_1_Surface_Temp': (self.start_address + 6828, self.start_address + 6829),
            'Ext_Probe_1_Pressure': (self.start_address + 6826, self.start_address + 6827),
            'Ext_Probe_1_Humidity': (self.start_address + 6824, self.start_address + 6825),
            'Ext_Probe_1_Tilt_Angle': (self.start_address + 6836, self.start_address + 6837),
            'Ext_Probe_1_Acceleration_X': (self.start_address + 6830, self.start_address + 6831),
            'Ext_Probe_1_Acceleration_Y': (self.start_address + 6832, self.start_address + 6834),
            'Ext_Probe_1_Acceleration_Z': (self.start_address + 6834, self.start_address + 6835)
        }

        # We'll store references to the file and CSV writer so we can close them or reopen them
        self.file = None
        self.writer = None

    def connect(self):
        """Establish connection to PQube3 with retry mechanism."""
        retries = 3
        for attempt in range(retries):
            if self.client.connect():
                print(f"Connected to PQube3 on attempt {attempt+1}")
                return
            print(f"Connection attempt {attempt+1} failed. Retrying...")
            time.sleep(2)
        raise ConnectionError("Failed to connect to PQube3 after retries.")

    def disconnect(self):
        """Close the connection to PQube3."""
        self.client.close()

    def read_float(self, start_register):
        """Read a 32-bit float from two consecutive registers."""
        try:
            response = self.client.read_holding_registers(start_register, count=2)
            if response.isError():
                print(f"Modbus error reading register {start_register}")
                return None
            raw = response.registers
            return self.convert_to_float(raw)
        except (ModbusException, ConnectionException) as e:
            print(f"Error reading register {start_register}: {e}")
            return None

    @staticmethod
    def convert_to_float(registers):
        """Convert two 16-bit registers to a floating point number."""
        import struct
        if len(registers) < 2:
            return None
        packed = struct.pack('>HH', registers[0], registers[1])
        return struct.unpack('>f', packed)[0]

    def open_log_file(self, date_str):
        """
        Helper function to open the CSV file for a given date in either append or write mode.
        We only write the header if the file does not exist (i.e., new file).
        """
        output_file = rf"C:\Users\mikhails.WISMAIN\Weizmann Institute Dropbox\Mikhail Svintsov\Quantum Circuits Lab\Fridges\Logs\LabLogs\{date_str}.csv"

        # Check if file already exists
        file_exists = os.path.exists(output_file)

        # If the file exists, open in append mode; otherwise, create a new file
        if file_exists:
            self.file = open(output_file, mode='a', newline='')
            self.writer = csv.writer(self.file)
        else:
            self.file = open(output_file, mode='w', newline='')
            self.writer = csv.writer(self.file)
            # Write header for new files
            self.writer.writerow(['Timestamp'] + list(self.registers.keys()))

        return output_file

    def log_data(self):
        """
        Continuously read and log data from PQube3, creating or appending to files daily.
        If the date changes, close the current file and open a new one for the new date.
        """
        current_date = datetime.datetime.now().strftime('%Y_%m_%d')
        # Open or append to today's log file
        self.open_log_file(current_date)

        try:
            while True:
                new_date = datetime.datetime.now().strftime('%Y_%m_%d')
                
                if new_date != current_date:
                    # The day changed - close current file and open a new one
                    self.file.close()
                    current_date = new_date
                    self.open_log_file(current_date)

                # Prepare row data
                row = [time.strftime('%Y-%m-%d %H:%M:%S')]
                for metric, (reg1, reg2) in self.registers.items():
                    if reg1 is not None:
                        value = self.read_float(reg1)
                    else:
                        value = 'N/A'
                    row.append(value)

                # Write row, flush to ensure data is on disk
                self.writer.writerow(row)
                self.file.flush()

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("Logging stopped.")
        finally:
            self.file.close()
            self.disconnect()

if __name__ == '__main__':
    pq_logger = PQube3Logger(host='132.77.44.166')  # Replace with actual IP
    pq_logger.connect()
    pq_logger.log_data()
