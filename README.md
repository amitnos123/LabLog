# LabLog
Monitoring station for the lab environment

## General Information
For a comprehensive overview of the device and its capabilities, please refer to the [Device_overview.pdf](./Device_overview.pdf).

## Usage Examples
Example workflows and use cases can be found in [Examples.pdf](./Examples.pdf).

## Dashboard Example
An example of a generated dashboard is available in [Dashboard_example.html](./Dashboard_example.html).

## Scripts and Files

1. **pq_edge_logger.py**  
   - Contains code to log data to a CSV file.

2. **pq_edge_logger_database.py**  
   - Contains code to log data to a QCODE database file.

3. **dashboard_v02.py**  
   - Generates an interactive dashboard for data visualization.

4. **2025_02_20.csv**  
   - Sample CSV data file.

## Getting Started

1. **Logging Data**  
   - Use `pq_edge_logger.py` to log device data to a CSV file.  
   - Use `pq_edge_logger_database.py` to log device data to a QCODE database.

2. **Creating a Dashboard**  
   - Run `dashboard_v02.py` to create a dashboard. This will generate or update an interactive visualization of your logged data.

3. **Viewing the Dashboard**  
   - Open `Dashboard_example.html` in your browser to see an example of the final dashboard layout and functionality.

Feel free to modify or extend these scripts according to your specific requirements. If you have any questions or issues, please consult the documentation in the provided PDF files or reach out to the repository maintainers.
