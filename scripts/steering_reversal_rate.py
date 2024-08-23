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

def extract_steering_angle_from_axes(message):
    """Extract steering angle from the 'axes' part of the message string."""
    match = re.search(r'axes:\s*\[([\-]?\d+\.?\d*)', message)
    return float(match.group(1)) if match else np.nan

def extract_time(message):
    """Extract and calculate total seconds from the message string."""
    match = re.search(r'secs:\s*(\d+)\s*nsecs:\s*(\d+)', message)
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

def find_stationary_points(angles):
    """Find stationary points using the method described in the paper."""
    theta_prime = np.zeros_like(angles)
    
    # Calculate the first derivative (scaled difference)
    for i in range(1, len(angles)):
        theta_prime[i] = angles[i] - angles[i-1]
    
    stationary_points = []
    
    for i in range(1, len(angles) - 1):
        # Condition 1: theta_prime is zero (stationary point)
        if theta_prime[i] == 0:
            stationary_points.append(i)
        # Condition 2: sign change in theta_prime (potential stationary point)
        elif np.sign(theta_prime[i]) != np.sign(theta_prime[i+1]):
            stationary_points.append(i)
    
    return stationary_points

def detect_reversals(angles, stationary_points, theta_min):
    """Detect both upwards and downwards reversals in the steering angle data."""
    def find_reversals(angles, stationary_points, theta_min):
        reversals = []
        k = 0
        for l in range(1, len(stationary_points)):
            start_idx = stationary_points[k]
            end_idx = stationary_points[l]

            if angles[end_idx] - angles[start_idx] >= theta_min:  # Upwards reversal
                reversals.append((start_idx, end_idx))
                k = l
            elif angles[end_idx] < angles[start_idx]:  # Move k to this point to search for next reversal
                k = l

        return reversals

    # Detect upwards reversals
    upwards_reversals = find_reversals(angles, stationary_points, theta_min)

    # Detect downwards reversals by applying the same algorithm to the negative of the angles
    downwards_reversals = find_reversals(-angles, stationary_points, theta_min)

    # Combine upwards and downwards reversals
    all_reversals = upwards_reversals + downwards_reversals

    return all_reversals

def plot_steering_angle(data, reversals, participant_id, condition, output_dir):
    """Plot steering angle with reversal markers and save with a filename based on the plot title."""
    plt.figure(figsize=(14, 8))

    # Use a pastel color palette
    steering_line, = plt.plot(data['TimeSeconds'], data['SteeringAngle'], marker='o', color='#89CFF0', markersize=4, label='Steering Wheel Angle')

    reversal_lines = []
    reversal_dots = []

    for start, end in reversals:
        if end < len(data):
            # Plot a line between the start and end of the reversal
            line, = plt.plot([data.iloc[start]['TimeSeconds'], data.iloc[end]['TimeSeconds']],
                             [data.iloc[start]['SteeringAngle'], data.iloc[end]['SteeringAngle']],
                             color='#FFB6C1', linewidth=2, zorder=5)
            reversal_lines.append(line)

            # Mark the start and end with circles
            dot_start = plt.scatter(data.iloc[start]['TimeSeconds'], data.iloc[start]['SteeringAngle'], color='#FF69B4', marker='o', s=100, zorder=6)
            dot_end = plt.scatter(data.iloc[end]['TimeSeconds'], data.iloc[end]['SteeringAngle'], color='#FF69B4', marker='o', s=100, zorder=6)
            reversal_dots.extend([dot_start, dot_end])

    # Add legend
    plt.legend([steering_line, reversal_lines[0], reversal_dots[0]], 
               ['Steering Wheel Angle', 'Reversal Line', 'Reversal Start/End'])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Steering Wheel Angle (radians)')
    title = f'Steering Wheel Angle with Reversals - Participant {participant_id}, Condition {condition}'
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    
    # Create the filename based on the plot title
    sanitized_title = re.sub(r'\W+', '_', title)  # Replace non-alphanumeric characters with underscores
    output_path = os.path.join(output_dir, f"{sanitized_title}.png")
    
    plt.savefig(output_path)
    plt.close()

def process_single_file(file_path, participant_id, condition, output_dir):
    """Process a single CSV file and return the total reversals, reversal rate, and session length."""
    data = load_data(file_path)
    if data is None:
        return 0, 0.0, "0:00"

    data['SteeringAngle'] = data['Message'].apply(extract_steering_angle_from_axes)
    data = data.dropna(subset=['SteeringAngle'])

    data['TotalSeconds'] = data['Message'].apply(extract_time)
    data['TimeSeconds'] = data['TotalSeconds'] - data['TotalSeconds'].min()  # Time in seconds from start

    if len(data['SteeringAngle']) > 9:
        sampling_rate = 100  # Hz
        cutoff_frequency = 0.6  # Hz
        data['FilteredSteeringAngle'] = low_pass_filter(data['SteeringAngle'].values, cutoff_frequency, sampling_rate)
    else:
        print("Data length is insufficient for filtering.")
        data['FilteredSteeringAngle'] = data['SteeringAngle']

    # Find stationary points in the filtered steering angle data
    stationary_points = find_stationary_points(data['FilteredSteeringAngle'].values)
    
    # Detect both upwards and downwards reversals using the improved method
    theta_min = 0.06  # Threshold for significant reversal (degrees, converted to radians if needed)
    reversals = detect_reversals(data['FilteredSteeringAngle'].values, stationary_points, theta_min)
    
    # Calculate the total duration of the session in seconds
    total_duration = data['TimeSeconds'].max() - data['TimeSeconds'].min()

    # Convert total_duration to minutes and seconds format
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    session_length = f"{minutes}:{seconds:02d}"
    
    # Calculate the reversal rate
    reversal_rate = len(reversals) / (total_duration / 60)  # Reversals per minute

    # Save the plot as a PNG file with a title-based filename
    plot_steering_angle(data, reversals, participant_id, condition, output_dir)
    
    return len(reversals), reversal_rate, session_length

def process_all_participants_data(data_logs_dir):
    """Process all participant data in the data_logs directory."""
    results = []

    for participant_dir in os.listdir(data_logs_dir):
        participant_path = os.path.join(data_logs_dir, participant_dir)
        if os.path.isdir(participant_path):
            participant_id = participant_dir.split('_')[1]
            for condition_dir in os.listdir(participant_path):
                condition_path = os.path.join(participant_path, condition_dir)
                if os.path.isdir(condition_path):
                    condition = condition_dir.split('_')[1]
                    # Process the specific CSV file for each participant and condition
                    csv_file = os.path.join(condition_path, '_Operator_InputDevices_joystick.csv')
                    if os.path.exists(csv_file):
                        total_reversals, reversal_rate, session_length = process_single_file(csv_file, participant_id, condition, condition_path)

                        # Save the results
                        results.append({
                            "Participant ID": participant_id,
                            "Condition": condition,
                            "Reversals": total_reversals,
                            "Reversal rate": reversal_rate,
                            "Session Length (min:sec)": session_length
                        })
    # Save the results to a main CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_logs_dir, 'steering_wheel_reversals_summary.csv'), index=False)

if __name__ == "__main__":
    data_logs_dir = os.path.join(os.path.dirname(__file__), '../data_logs')
    process_all_participants_data(data_logs_dir)
