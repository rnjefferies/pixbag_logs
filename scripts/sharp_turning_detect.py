import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import butter, filtfilt

def load_data(file_path):
    """Load CSV data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def extract_steering_angle(message):
    """Extract steering angle from the message string."""
    match = re.search(r'steeringWheelAngle:\s*([\-]?\d+\.?\d*)', message)
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

def plot_turning_events(data, turning_events, participant_id, condition, output_dir):
    """Plot steering angle with sharp turning events highlighted."""
    plt.figure(figsize=(14, 8))

    # Use a pastel color palette
    angle_line, = plt.plot(data['TimeSeconds'], data['SteeringAngle'], marker='o', color='#89CFF0', markersize=4, label='Steering Angle')

    reversal_lines = []
    reversal_dots = []

    for start, end in turning_events:
        if end < len(data):
            # Plot a line between the start and end of the turn
            line, = plt.plot([data.iloc[start]['TimeSeconds'], data.iloc[end]['TimeSeconds']],
                             [data.iloc[start]['SteeringAngle'], data.iloc[end]['SteeringAngle']],
                             color='#FFB6C1', linewidth=2, zorder=5)
            reversal_lines.append(line)

            # Mark the start and end with circles
            dot_start = plt.scatter(data.iloc[start]['TimeSeconds'], data.iloc[start]['SteeringAngle'], color='#FF69B4', marker='o', s=100, zorder=6)
            dot_end = plt.scatter(data.iloc[end]['TimeSeconds'], data.iloc[end]['SteeringAngle'], color='#FF69B4', marker='o', s=100, zorder=6)
            reversal_dots.extend([dot_start, dot_end])

    # Add legend
    plt.legend([angle_line, reversal_lines[0], reversal_dots[0]], 
               ['Steering Angle', 'Sharp Turn Line', 'Sharp Turn Start/End'])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Steering Angle (radians)')
    title = f'Sharp Turn Events - Participant {participant_id}, Condition {condition}'
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    
    # Create the filename based on the plot title
    sanitized_title = re.sub(r'\W+', '_', title)  # Replace non-alphanumeric characters with underscores
    output_path = os.path.join(output_dir, f"{sanitized_title}.png")
    
    plt.savefig(output_path)
    plt.close()

def process_single_file(file_path, participant_id, condition, output_dir):
    """Process a single CSV file and return detected sharp turn events."""
    data = load_data(file_path)
    if data is None:
        return

    data['SteeringAngle'] = data['Message'].apply(extract_steering_angle)
    data['TotalSeconds'] = data['Message'].apply(extract_time)
    data = data.dropna(subset=['SteeringAngle', 'TotalSeconds'])

    # Normalize time so that it starts from zero
    data['TimeSeconds'] = data['TotalSeconds'] - data['TotalSeconds'].min()

    if len(data) > 9:
        # Filter and detect sharp turns
        sampling_rate = 10  # Hz
        cutoff_frequency = 0.6  # Hz
        data['FilteredSteeringAngle'] = low_pass_filter(data['SteeringAngle'].values, cutoff_frequency, sampling_rate)
        stationary_points = find_stationary_points(data['FilteredSteeringAngle'].values)
        theta_min = 0.06  # radians, minimum steering angle change for sharp turns
        turning_events = detect_reversals(data['FilteredSteeringAngle'].values, stationary_points, theta_min)

        # Plot detected events
        plot_turning_events(data, turning_events, participant_id, condition, output_dir)

        return len(turning_events)
    else:
        print("Data length is insufficient for analysis.")
        return 0

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
                    csv_file = os.path.join(condition_path, '_Operator_VehicleBridge_vehicle_data.csv')
                    if os.path.exists(csv_file):
                        turning_events = process_single_file(csv_file, participant_id, condition, condition_path)

                        # Save the results
                        results.append({
                            "Participant ID": participant_id,
                            "Condition": condition,
                            "Sharp Turn Events": turning_events
                        })

    # Save the results to a main CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_logs_dir, 'sharp_turn_summary.csv'), index=False)

if __name__ == "__main__":
    data_logs_dir = os.path.join(os.path.dirname(__file__), '../data_logs')
    process_all_participants_data(data_logs_dir)
