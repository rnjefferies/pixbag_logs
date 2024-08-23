import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import re
from datetime import datetime

def load_data(file_path):
    """Load CSV data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def extract_longitudinal_speed(message):
    """Extract longitudinal speed from the message string."""
    match = re.search(r'longitudinalSpeed:\s*([\-]?\d+\.?\d*)', message)
    return float(match.group(1)) if match else np.nan

def extract_time(message):
    """Extract and calculate total seconds from the message string."""
    match = re.search(r'stamp:\s*secs:\s*(\d+)\s*nsecs:\s*(\d+)', message)
    if match:
        secs = int(match.group(1))
        nsecs = int(match.group(2))
        return secs + nsecs / 1e9
    return np.nan

def low_pass_filter(data, cutoff_frequency, sampling_rate):
    """Apply a low-pass filter to the data."""
    if len(data) <= 9:
        raise ValueError("Data length is too short for the filter.")
    
    nyquist_frequency = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def find_stationary_points(speeds):
    """Find stationary points in the speed data."""
    speed_prime = np.zeros_like(speeds)
    
    # Calculate the first derivative (scaled difference)
    for i in range(1, len(speeds)):
        speed_prime[i] = speeds[i] - speeds[i-1]
    
    stationary_points = []
    
    for i in range(1, len(speeds) - 1):
        # Condition 1: speed_prime is zero (stationary point)
        if speed_prime[i] == 0:
            stationary_points.append(i)
        # Condition 2: sign change in speed_prime (potential stationary point)
        elif np.sign(speed_prime[i]) != np.sign(speed_prime[i+1]):
            stationary_points.append(i)
    
    return stationary_points

def detect_reversals(speeds, stationary_points, theta_min):
    """Detect both upwards and downwards reversals in the speed data."""
    def find_reversals(speeds, stationary_points, theta_min):
        reversals = []
        k = 0
        for l in range(1, len(stationary_points)):
            start_idx = stationary_points[k]
            end_idx = stationary_points[l]

            if speeds[end_idx] - speeds[start_idx] >= theta_min:  # Upwards reversal
                reversals.append((start_idx, end_idx))
                k = l
            elif speeds[end_idx] < speeds[start_idx]:  # Move k to this point to search for next reversal
                k = l

        return reversals

    # Detect upwards reversals
    upwards_reversals = find_reversals(speeds, stationary_points, theta_min)

    # Detect downwards reversals by applying the same algorithm to the negative of the speeds
    downwards_reversals = find_reversals(-speeds, stationary_points, theta_min)

    # Combine upwards and downwards reversals
    all_reversals = upwards_reversals + downwards_reversals

    return all_reversals

def plot_longitudinal_speed(data, reversals, output_path):
    """Plot longitudinal speed with reversal markers highlighted as lines with circles at start and end points."""
    plt.figure(figsize=(14, 8))

    # Use pastel colors
    speed_line, = plt.plot(data['TimeSeconds'], data['LongitudinalSpeed'], marker='o', color='#89CFF0', markersize=4, label='Longitudinal Speed')

    reversal_lines = []
    reversal_dots = []

    for start, end in reversals:
        if end < len(data):
            # Plot a line between the start and end of the reversal
            line, = plt.plot([data.iloc[start]['TimeSeconds'], data.iloc[end]['TimeSeconds']],
                             [data.iloc[start]['LongitudinalSpeed'], data.iloc[end]['LongitudinalSpeed']],
                             color='#FFB6C1', linewidth=2, zorder=5)
            reversal_lines.append(line)

            # Mark the start and end with circles
            dot_start = plt.scatter(data.iloc[start]['TimeSeconds'], data.iloc[start]['LongitudinalSpeed'], color='#FF69B4', marker='o', s=100, zorder=6)
            dot_end = plt.scatter(data.iloc[end]['TimeSeconds'], data.iloc[end]['LongitudinalSpeed'], color='#FF69B4', marker='o', s=100, zorder=6)
            reversal_dots.extend([dot_start, dot_end])

    # Add legend
    plt.legend([speed_line, reversal_lines[0], reversal_dots[0]], 
               ['Longitudinal Speed', 'Reversal Line', 'Reversal Start/End'])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Longitudinal Speed (m/s)')
    plt.title('Longitudinal Speed with Reversals Highlighted')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_all_participants_data(data_logs_dir):
    """Process all participant data in the data_logs directory."""
    results = []

    for participant_dir in os.listdir(data_logs_dir):
        participant_path = os.path.join(data_logs_dir, participant_dir)
        if os.path.isdir(participant_path):
            for condition_dir in os.listdir(participant_path):
                condition_path = os.path.join(participant_path, condition_dir)
                if os.path.isdir(condition_path):
                    # Process the specific CSV file for each participant and condition
                    csv_file = os.path.join(condition_path, '_Operator_VehicleBridge_vehicle_data.csv')
                    if os.path.exists(csv_file):
                        total_reversals, reversal_rate, session_length = process_single_file(csv_file)

                        # Save the results
                        results.append({
                            "Participant ID": participant_dir.split('_')[1],
                            "Condition": condition_dir.split('_')[1],
                            "Reversals": total_reversals,
                            "Reversal rate": reversal_rate,
                            "Session Length (min:sec)": session_length
                        })
    # Save the results to a main CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_logs_dir, 'speed_reversals_summary.csv'), index=False)

def process_single_file(file_path):
    """Process a single CSV file and return the total reversals, reversal rate, and session length."""
    data = load_data(file_path)
    if data is None:
        return 0, 0.0, "0:00"

    data['LongitudinalSpeed'] = data['Message'].apply(extract_longitudinal_speed)
    data = data.dropna(subset=['LongitudinalSpeed'])

    data['TotalSeconds'] = data['Message'].apply(extract_time)
    data['TimeSeconds'] = data['TotalSeconds'] - data['TotalSeconds'].min()  # Time in seconds from start

    if len(data['LongitudinalSpeed']) > 9:
        sampling_rate = 10  # Hz
        cutoff_frequency = 0.6  # Hz
        data['FilteredLongitudinalSpeed'] = low_pass_filter(data['LongitudinalSpeed'].values, cutoff_frequency, sampling_rate)
    else:
        print("Data length is insufficient for filtering.")
        data['FilteredLongitudinalSpeed'] = data['LongitudinalSpeed']

    # Find stationary points in the filtered speed data
    stationary_points = find_stationary_points(data['FilteredLongitudinalSpeed'].values)
    
    # Detect both upwards and downwards reversals using the improved method
    theta_min = 0.06  # Threshold for significant reversal
    reversals = detect_reversals(data['FilteredLongitudinalSpeed'].values, stationary_points, theta_min)
    
    # Calculate the total duration of the session in seconds
    total_duration = data['TimeSeconds'].max() - data['TimeSeconds'].min()

    # Convert total_duration to minutes and seconds format
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    session_length = f"{minutes}:{seconds:02d}"
    
    # Calculate the reversal rate
    reversal_rate = len(reversals) / (total_duration / 60)  # Reversals per minute

    # Save the plot as a PNG next to the CSV file
    plot_longitudinal_speed(data, reversals, output_path=file_path.replace('.csv', '.png'))
    
    return len(reversals), reversal_rate, session_length

if __name__ == "__main__":
    data_logs_dir = os.path.join(os.path.dirname(__file__), '../data_logs')
    process_all_participants_data(data_logs_dir)
