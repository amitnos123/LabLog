import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
from scipy.stats import zscore
from scipy.signal import spectrogram
from scipy.interpolate import griddata

# ----------------------------------------------------------------------
# 1) HELPER FUNCTIONS (same as your current script)
# ----------------------------------------------------------------------

def remove_outliers_zscore(df, metric, threshold=3):
    z_scores = zscore(df[metric], nan_policy='omit')  
    # return df[abs(z_scores) < threshold]
    return df

def remove_outliers_for_metrics(df, metrics, threshold=3):
    for metric in metrics:
        df = remove_outliers_zscore(df, metric, threshold)
    return df

def compute_fft(time_series, values):
    """
    Returns freq (Hz) and FFT magnitude.
    Assumes approximate uniform sampling (mean dt).
    """
    threshold = 3
    z_scores = zscore(values, nan_policy='omit')
    val = values[abs(z_scores) < threshold]
    ts = time_series[abs(z_scores) < threshold]
    time_seconds = (ts - ts.iloc[0]).dt.total_seconds().values
    if len(time_seconds) < 2:
        return np.array([]), np.array([])
    
    dt = np.mean(np.diff(time_seconds))
    if dt == 0:
        return np.array([]), np.array([])

    data_detrended = val - np.mean(val)
    fft_vals = np.fft.fft(data_detrended)
    freqs = np.fft.fftfreq(len(data_detrended), d=dt)

    # Only non-negative freq
    mask = freqs >= 0
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]
    fft_mag = np.abs(fft_vals)
    return freqs, fft_mag

def compute_spectrogram(time_series, values):
    """
    Compute a spectrogram (time-frequency).
    """
    time_seconds = (time_series - time_series.iloc[0]).dt.total_seconds().values
    if len(time_seconds) < 2:
        return np.array([]), np.array([]), np.array([[]])

    dt = np.mean(np.diff(time_seconds))
    fs = 1.0 / dt  # sampling frequency
    
    f, t, Sxx = spectrogram(values, fs=fs, nperseg=256, noverlap=128)

    x, y = np.indices(Sxx.shape)

    # Mask for valid (non-NaN) points
    valid = ~np.isnan(Sxx)

    # Interpolate using the valid points
    Sxx_interp = griddata(
        (x[valid], y[valid]),       # Coordinates of valid data
        Sxx[valid],                 # Valid values
        (x, y),                     # Grid to interpolate over
        method='linear'
    )


    return f, t, Sxx_interp

def add_metric_plots(fig, df_metric, metric_name, row_idx):
    """
    1) FFT (col=1, log–log)
    2) Spectrogram (col=2, x-axis in real time)
    3) Time series (col=3)
    4) Histogram (col=4)
    """
    # A) Compute FFT
    freqs, fft_mag = compute_fft(df_metric["Timestamp"], df_metric[metric_name])
    
    # B) Compute Spectrogram
    f_sg, t_sg, Sxx = compute_spectrogram(df_metric["Timestamp"], df_metric[metric_name])
    Sxx_db = 10 * np.log10(Sxx + 1e-20)  # dB scale

    # Convert spectrogram t_sg (which is in seconds) to actual datetimes
    if len(df_metric["Timestamp"]) > 0:
        t_start = df_metric["Timestamp"].iloc[0]
        # Convert from seconds to timedelta, then add to the start
        spectrogram_time = t_start + pd.to_timedelta(t_sg, unit='s')
    else:
        # Fallback if no data
        spectrogram_time = t_sg

    # (1) FFT in col=1 => log–log
    fft_trace = go.Scatter(
        x=freqs,
        y=fft_mag,
        mode='lines',
        name=f"{metric_name} FFT (log–log)",
    )
    fig.add_trace(fft_trace, row=row_idx, col=1)
    fig.update_xaxes(type="log", row=row_idx, col=1)
    fig.update_yaxes(type="log", row=row_idx, col=1)

    # (2) Spectrogram in col=2 => no colorbar
    # Use spectrogram_time on the x-axis and frequencies on the y-axis
    spec_trace = go.Heatmap(
        x=spectrogram_time,
        y=f_sg,
        z=Sxx_db,
        colorscale='Viridis',
        zsmooth='best',
        showscale=False,  # no colorbar
        name=f"{metric_name} Spectrogram"
    )
    fig.add_trace(spec_trace, row=row_idx, col=2)

    # Make sure the x-axis in col=2 is treated as a date axis
    fig.update_xaxes(type="date", row=row_idx, col=2)

    # (3) Time series in col=3
    ts_trace = go.Scatter(
        x=df_metric["Timestamp"],
        y=df_metric[metric_name],
        mode='lines',
        name=metric_name
    )
    if metric_name == 'Frequency':
        fig.update_yaxes(range=[49.8, 50.13], row=row_idx, col=3)
    fig.add_trace(ts_trace, row=row_idx, col=3)
    


    # (4) Histogram in col=4
    if metric_name == 'Frequency':
        hist_trace = go.Histogram(
            y=df_metric[metric_name][df_metric[metric_name] > 49.8],
            marker=dict(color="yellow"),
            name=f"{metric_name} Dist",
            opacity=0.75
        )
    else:
        hist_trace = go.Histogram(
            y=df_metric[metric_name],
            marker=dict(color="yellow"),
            name=f"{metric_name} Dist",
            opacity=0.75
        )
    fig.add_trace(hist_trace, row=row_idx, col=4)

    # Enable y-axis tick labels for columns 1, 2, and 3
    fig.update_yaxes(showticklabels=True, row=row_idx, col=1)
    fig.update_yaxes(showticklabels=True, row=row_idx, col=2)
    fig.update_yaxes(showticklabels=True, row=row_idx, col=3)

    # Disable y-axis tick labels for column 4 (histogram)
    fig.update_yaxes(showticklabels=False, row=row_idx, col=4)



def absolut_value_accelleration(df):
    # Filter out outliers in X, Y, Z
    df_clean = remove_outliers_for_metrics(
        df,
        ['Ext_Probe_1_Acceleration_X',
         'Ext_Probe_1_Acceleration_Y',
         'Ext_Probe_1_Acceleration_Z']
    )
    time = df_clean["Timestamp"]
    abs_acc = np.sqrt(
        df_clean["Ext_Probe_1_Acceleration_X"]**2
        + df_clean["Ext_Probe_1_Acceleration_Y"]**2
        + df_clean["Ext_Probe_1_Acceleration_Z"]**2
    )
    return time, abs_acc


# ----------------------------------------------------------------------
# 2) FIGURE-GENERATION FUNCTION
# ----------------------------------------------------------------------

def create_figure():
    # Your data selection
    data_start = "2025_01_07"  # or another date
    
    folder_path = (
        r"C:\Users\mikhails.WISMAIN\Weizmann Institute Dropbox"
        r"\Mikhail Svintsov\Quantum Circuits Lab\Fridges\Logs\LabLogs"
    )
    
    # Get all CSV file paths
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Filter files by date
    filtered_files = [
        file for file in csv_files if os.path.basename(file) >= f"{data_start}.csv"
    ]

    if not filtered_files:
        # if no files, return an empty figure or something minimal
        return go.Figure()

    df_list = [pd.read_csv(file) for file in filtered_files]
    df = pd.concat(df_list, ignore_index=True)

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    # Example filter by “last day”
    previous_day = (datetime.now() - timedelta(days=2)).date()
    df = df[df["Timestamp"].dt.date > previous_day]

    # The columns you are using
    metrics = df.columns[1:-4]
    metrics_name = [
        'Voltage, V', 
        'Voltage Frequency, Hz', 
        'Surface Temperature, °C', 
        'Atmospheric Pressure, mbar', 
        'Humidity, %', 
        'Acceleration, m/s²'
    ]

    # Prepare the figure layout
    nrows = len(metrics) + 1
    fig = make_subplots(
        rows=nrows,
        cols=4,
        shared_xaxes=False,
        shared_yaxes=True,
        column_widths=[0.10, 0.35, 0.35, 0.10],
        horizontal_spacing=0.04,
        subplot_titles=np.array([
            ["Frequency distribution"] 
            + ["Spectrogram"] 
            + [f"{metric}"] 
            + ["Histogram"] 
            for metric in metrics_name
        ]).flatten()
    )

    # (A) Standard metrics
    for i, metric in enumerate(metrics, start=1):
        df_filtered = remove_outliers_zscore(df, metric)
        add_metric_plots(fig, df_filtered, metric, row_idx=i)

    # (B) Absolute Value Acceleration (last row)
    acc_time, acc_value = absolut_value_accelleration(df)
    acc_df = pd.DataFrame({"Timestamp": acc_time, "AbsAcc": acc_value})
    row_idx = nrows
    add_metric_plots(fig, acc_df, "AbsAcc", row_idx=row_idx)

    # Final layout updates
    fig.update_layout(
        title="Lab's environment scan (V.0.2)",
        template="plotly_dark",
        height=300 * nrows,
        showlegend=False
    )
    return fig


# ----------------------------------------------------------------------
# 3) DASH APP
# ----------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div([
    # Interval component, e.g. update every 60,000 ms (1 minute).
    # Adjust interval to your needs (in milliseconds).
    dcc.Interval(
        id='interval-component',
        interval=10*60*1000,
        n_intervals=0
    ),
    dcc.Graph(id='live-graph')
])

# Callback to update the figure
@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_figure(n):
    """
    This callback is triggered every `interval` milliseconds
    (set above). Each time, we regenerate the figure from scratch,
    reading the latest CSV data, removing outliers, etc.
    """
    fig = create_figure()
    return fig

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8055, debug=False)
