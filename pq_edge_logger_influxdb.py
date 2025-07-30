import csv
import time
import datetime
import os
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException, ConnectionException
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import scipy.fft as fft
import numpy as np

class PQube3Logger:
    def __init__(self, host, port=502, interval=0.08, buffer_length=20, fft_input_length=1000):
        self.client = ModbusTcpClient(host, port=port, timeout=3)  # Set timeout
        self.interval = interval
        self.buffer_length = buffer_length 
        self.start_address = 7000  # The Modbus start address
        self.fft_input_length = fft_input_length # Length of FFT input in intervals. 1000 intervals means 8.3 minutes of data at 0.5s intervals.
        self.fft_input = [0 for _ in range(self.fft_input_length)]  # Initialize FFT input array
        self.fft_output = []  
        self.fft_freq = [] # Frequency bins for FFT, assuming uniform sampling
        self.registers = {
            'L1-E Voltage': (self.start_address + 0, self.start_address + 1),
            'Frequency': (self.start_address + 26, self.start_address + 27),
            'Ext_Probe_1_Surface_Temp': (self.start_address + 6828, self.start_address + 6829),
            'Ext_Probe_1_Pressure': (self.start_address + 6826, self.start_address + 6827),
            'Ext_Probe_1_Humidity': (self.start_address + 6824, self.start_address + 6825),
            'Ext_Probe_1_Tilt_Angle': (self.start_address + 6836, self.start_address + 6837),
            'Ext_Probe_1_Acceleration_X': (self.start_address + 6830, self.start_address + 6831),
            'Ext_Probe_1_Acceleration_Y': (self.start_address + 6832, self.start_address + 6834),
            'Ext_Probe_1_Acceleration_Z': (self.start_address + 6834, self.start_address + 6835),
            'L1_Voltage_Harmonic_H1': (self.start_address + 2232, self.start_address + 2233),
            'L1_Voltage_Harmonic_H2': (self.start_address + 2234, self.start_address + 2235),
            'L1_Voltage_Harmonic_H3': (self.start_address + 2236, self.start_address + 2237),
            'L1_Voltage_Harmonic_H4': (self.start_address + 2238, self.start_address + 2239),
            'L1_Voltage_Harmonic_H5': (self.start_address + 2240, self.start_address + 2241),
            'L1_Voltage_Harmonic_H6': (self.start_address + 2242, self.start_address + 2243),
            'L1_Voltage_Harmonic_H7': (self.start_address + 2244, self.start_address + 2245),
            'L1_Voltage_Harmonic_H8': (self.start_address + 2246, self.start_address + 2247),
            'L1_Voltage_Harmonic_H9': (self.start_address + 2248, self.start_address + 2249),
            'L1_Voltage_Harmonic_H10': (self.start_address + 2250, self.start_address + 2251),
            'L1_Voltage_THD': (self.start_address + 192, self.start_address + 193),
        }

        # InfluxDB client setup
        self.influxdb_bucket = "influxdb_bucket"
        self.influxdb_org = "influxdb_org"
        self.influxdb_token = "influxdb_token"
        # Store the URL of your InfluxDB instance
        self.influxdb_url="influxdb_url"
        self.influxdb_client = influxdb_client.InfluxDBClient(
            url=self.influxdb_url,
            token=self.influxdb_token,
            org=self.influxdb_org
        )
        self.write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)

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

    def get_init_data_frame(self, date_str):
        """        Initialize a DataFrame for the day's data, either loading existing data or creating a new one."""
        # Define the output file path
        self.output_file = rf"C:\Users\rosengrp\Weizmann Institute Dropbox\Serge Rosenblum\Quantum Circuits Lab\Fridges\Logs\LabLogs\{date_str}.gzip"

        # Check if file already exists
        file_exists = os.path.exists(self.output_file)
        if file_exists:
            return pd.read_parquet(self.output_file, engine='pyarrow')
        else:
            return pd.DataFrame(columns=['Timestamp'] + list(self.registers.keys()))

    def log_data(self):
        """
        Continuously read and log data from PQube3, creating or appending to files daily.
        If the date changes, close the current file and open a new one for the new date.
        """
        current_date = datetime.datetime.now().strftime('%Y_%m_%d')
        # Open or append to today's log file
        df = self.get_init_data_frame(current_date)

        buffer_count = 0

        try:
            while True:
                # Prepare row data
                row = [time.strftime('%Y-%m-%d %H:%M:%S')]
                for metric, (reg1, reg2) in self.registers.items():
                    if reg1 is not None:
                        value = self.read_float(reg1)
                    else:
                        value = 'N/A'
                    row.append(value)
                # Append the row to the DataFrame
                df.loc[len(df)] = row

                # self.fft_input.append(row[1])
                # if len(self.fft_input) > self.fft_input_length:
                #     # Remove the oldest entry if we exceed the length
                #     self.fft_input.pop(0)
                #     self.fft_freq = fft.fftfreq(len(self.fft_input), d=self.interval)
                #     self.fft_output = fft.fft(self.fft_input)
                # else:
                #     print("FFT input not yet full, skipping FFT calculation. FFT input length:", len(self.fft_input))

                buffer_count += 1
                # Check if buffer count has reached the buffer length to write to file
                if buffer_count >= self.buffer_length:
                    # Reset buffer count
                    buffer_count = 0
                    # Save DataFrame to Parquet file
                    df.to_parquet(self.output_file)
                
                # Write to InfluxDB
                try:
                    points = []

                    measurement_name = "pq_edge"
                    points.append(influxdb_client.Point(measurement_name).field("L1-E Voltage", row[1]))
                    points.append(influxdb_client.Point(measurement_name).field("Frequency", row[2]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Surface_Temp", row[3]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Pressure", row[4]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Humidity", row[5]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Tilt_Angle", row[6]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Acceleration_X", row[7]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Acceleration_Y", row[8]))
                    points.append(influxdb_client.Point(measurement_name).field("Ext_Probe_1_Acceleration_Z", row[9]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H1", row[10]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H2", row[11]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H3", row[12]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H4", row[13]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H5", row[14]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_THD", row[15]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H6", row[16]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H7", row[17]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H8", row[18]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H9", row[19]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_Harmonic_H10", row[19]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Voltage_THD", row[20]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Flicker_P_inst", row[21]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Flicker_P_st", row[22]))
                    points.append(influxdb_client.Point(measurement_name).field("L1_Flicker_P_lt", row[23]))

                    # if len(self.fft_input) >= self.fft_input_length:
                    #     # Add FFT output data
                    #     for i in range(len(self.fft_output)):
                    #         points.append(influxdb_client.Point(measurement_name).tag("bin", str(i)).field("FFT_Output_Real", self.fft_output[i].real).field("FFT_Output_Imag", self.fft_output[i].imag).field("FFT_Frequency", self.fft_freq[i]))
                            
                    # Send all points in one request
                    self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=points)
                except Exception as e:
                    print(f"Write failed to InfluxDB: {e}")

                new_date = datetime.datetime.now().strftime('%Y_%m_%d')
                
                if new_date != current_date:
                    # Reset buffer count
                    buffer_count = 0
                    # Save DataFrame to Parquet file
                    df.to_parquet(self.output_file)
                    # The day changed - close current file and open a new one
                    df = self.get_init_data_frame(current_date)
                    current_date = new_date

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("Logging stopped.")
        finally:
            df.to_parquet(self.output_file, compression='gzip')

if __name__ == '__main__':
    pq_logger = PQube3Logger(host='PQube3LoggerAddress')  # Replace with actual IP
    pq_logger.connect()
    pq_logger.log_data()
