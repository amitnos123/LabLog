import time
import datetime
import struct
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException, ConnectionException

import qcodes as qc
from qcodes.dataset.sqlite.database import initialise_or_create_database_at
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id


class PQube3QcodesLogger:
    def __init__(self, host, port=502, interval=0.1, db_path='PQube3_Logger.db'):
        self.client = ModbusTcpClient(host, port=port, timeout=3)
        self.interval = interval
        self.start_address = 7000

        # Define the registers you want to log
        self.registers = {
            'L1_E_Voltage': (self.start_address + 0, self.start_address + 1),
            'Frequency': (self.start_address + 26, self.start_address + 27),
            'Ext_Probe_1_Surface_Temp': (self.start_address + 6828, self.start_address + 6829),
            'Ext_Probe_1_Pressure': (self.start_address + 6826, self.start_address + 6827),
            'Ext_Probe_1_Humidity': (self.start_address + 6824, self.start_address + 6825),
            'Ext_Probe_1_Tilt_Angle': (self.start_address + 6836, self.start_address + 6837),
            'Ext_Probe_1_Acceleration_X': (self.start_address + 6830, self.start_address + 6831),
            'Ext_Probe_1_Acceleration_Y': (self.start_address + 6832, self.start_address + 6833),
            'Ext_Probe_1_Acceleration_Z': (self.start_address + 6834, self.start_address + 6835),
        }

        # Define units for each register (adjust these to match your setup)
        self.units = {
            'L1_E_Voltage': 'V',        # Volts
            'Frequency': 'Hz',
            'Ext_Probe_1_Surface_Temp': 'C',   # Celsius
            'Ext_Probe_1_Pressure': 'mbar',    # Pressure unit example
            'Ext_Probe_1_Humidity': '%',       # Percentage
            'Ext_Probe_1_Tilt_Angle': 'deg',   # Degrees
            'Ext_Probe_1_Acceleration_X': 'm/s^2',
            'Ext_Probe_1_Acceleration_Y': 'm/s^2',
            'Ext_Probe_1_Acceleration_Z': 'm/s^2',
        }

        # QCoDeS database path
        self.db_path = db_path

    def connect(self):
        """Establish connection to PQube3 with a simple retry mechanism."""
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
        """Read a 32-bit float from two consecutive registers via Modbus."""
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
        if len(registers) < 2:
            return None
        packed = struct.pack('>HH', registers[0], registers[1])
        return struct.unpack('>f', packed)[0]

    def log_data_to_qcodes(self):
        """
        Continuously read and log data from PQube3 into a QCoDeS database,
        using time in seconds as the independent (setpoint) parameter.
        """

        # 1) Initialize or create the database
        initialise_or_create_database_at(self.db_path)

        # 2) Load or create an experiment
        exp = load_or_create_experiment(
            experiment_name='PQube3_Logging',
            sample_name='MyPQube3Sample'
        )

        # 3) Set up a Measurement
        meas = Measurement(exp=exp)

        # Register 'time_s' as our independent parameter (setpoint)
        meas.register_custom_parameter(
            "time_s",
            paramtype="numeric",
            unit="s"
        )

        # Register each PQube3 metric as dependent on time_s
        for param_name in self.registers.keys():
            # Look up the associated unit from self.units
            unit_str = self.units.get(param_name, "")
            meas.register_custom_parameter(
                param_name,
                paramtype="numeric",
                setpoints=("time_s",),
                unit=unit_str
            )

        try:
            with meas.run() as datasaver:
                while True:
                    # Our independent setpoint
                    current_time = time.time()

                    # Read all PQube3 registers
                    row_data = {}
                    for reg_name, (reg_start, _) in self.registers.items():
                        value = self.read_float(reg_start)
                        row_data[reg_name] = value

                    # Prepare the row of data for datasaver
                    data_to_save = [("time_s", current_time)]
                    data_to_save += [
                        (reg_name, row_data[reg_name]) 
                        for reg_name in self.registers.keys()
                    ]

                    # Save data to the QCoDeS database
                    datasaver.add_result(*data_to_save)

                    # Wait for the desired interval
                    time.sleep(self.interval)

        except KeyboardInterrupt:
            print("Logging stopped by user.")
        finally:
            self.disconnect()


if __name__ == '__main__':
    pq_logger = PQube3QcodesLogger(host='132.77.44.166')
    pq_logger.connect()
    pq_logger.log_data_to_qcodes()
