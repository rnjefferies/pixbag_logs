import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

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

def extract_speed(message):
    """Extract speed from the message string."""
    match = re.search(r'longitudinalSpeed:\s*([\-]?\d+\.?\d*)', message)
    return float(match.group(1)) if match else np.nan

def extract_time(message):
    """Extract and calculate total seconds from the message string."""
    match = re.search(r'secs:\s*(\d+)\s*nsecs:\s*(\d+)', message)
    if match:
        secs = int(match.group(1))
        nsecs = int(match.group(2))
        return secs + nsecs / 1e9
    return np.nan

def compute_position(data, sampling_rate):
    """Compute the 2D position based on speed and steering angle."""
    # Initialize position and heading
    x, y = 0.0, 0.0
    heading = 0.0  # Assume starting heading is 0 radians (facing right on a 2D plot)
    
    positions = [(x, y)]
    
    for i in range(1, len(data)):
        # Time delta
        dt = data['TimeSeconds'].iloc[i] - data['TimeSeconds'].iloc[i - 1]
        
        # Compute change in heading based on steering angle
        steering_angle = data['SteeringAngle'].iloc[i]
        heading += (steering_angle * dt)
        
        # Compute displacement
        speed = data['Speed'].iloc[i]
        dx = speed * np.cos(heading) * dt
        dy = speed * np.sin(heading) * dt
        
        # Update position
        x += dx
        y += dy
        
        positions.append((x, y))
    
    return np.array(positions)

def detect_hard_braking_events(data, deceleration_threshold, duration_threshold):
    """Detect hard braking events based on a given deceleration threshold and a minimum duration."""
    events = []
    event_start = None

    for i in range(1, len(data)):
        speed_change = data['Speed'].iloc[i - 1] - data['Speed'].iloc[i]
        time_change = data['TimeSeconds'].iloc[i] - data['TimeSeconds'].iloc[i - 1]

        if time_change > 0:
            deceleration = speed_change / time_change

            if deceleration >= deceleration_threshold:
                if event_start is None:
                    event_start = data['TimeSeconds'].iloc[i - 1]
            else:
                if event_start is not None:
                    # Check if the event lasted long enough to be considered
                    if (data['TimeSeconds'].iloc[i - 1] - event_start) >= duration_threshold:
                        events.append(i)
                    event_start = None

    # Handle case where the event continues till the end
    if event_start is not None and (data['TimeSeconds'].iloc[-1] - event_start) >= duration_threshold:
        events.append(len(data) - 1)

    return events

def plot_route_with_heatmap(positions, speeds, events, participant_id, condition, output_dir):
    """Plot the 2D route taken with speed heatmap and hard braking events."""
    plt.figure(figsize=(14, 8))
    
    # Create a colormap based on speed
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())

    # Create a collection of line segments with colors based on speed
    segments = np.array([positions[:-1], positions[1:]]).transpose(1, 0, 2)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(speeds)
    lc.set_linewidth(2)

    # Plot the route with heatmap
    plt.gca().add_collection(lc)
    plt.colorbar(lc, label='Speed (km/h)')

    # Mark hard braking events
    if events:
        plt.scatter(positions[events, 0], positions[events, 1], color='#FF69B4', marker='x', s=100, zorder=5, label='Hard Braking Event')

    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    title = f'2D Route Map with Speed Heatmap - Participant {participant_id}, Condition {condition}'
    plt.title(title)
    plt.grid(True)
    
    plt.legend()
    plt.tight_layout()
    
    # Create the filename based on the plot title
    sanitized_title = re.sub(r'\W+', '_', title)  # Replace non-alphanumeric characters with underscores
    output_path = os.path.join(output_dir, f"{sanitized_title}.png")
    
    plt.savefig(output_path)
    plt.close()

def process_single_file(file_path, participant_id, condition, output_dir):
    """Process a single CSV file to compute and plot the route taken with speed heatmap and hard braking events."""
    data = load_data(file_path)
    if data is None:
        return

    data['SteeringAngle'] = data['Message'].apply(extract_steering_angle)
    data['Speed'] = data['Message'].apply(extract_speed)
    data['TotalSeconds'] = data['Message'].apply(extract_time)
    data = data.dropna(subset=['SteeringAngle', 'Speed', 'TotalSeconds'])

    # Normalize time so that it starts from zero
    data['TimeSeconds'] = data['TotalSeconds'] - data['TotalSeconds'].min()

    if len(data) > 9:
        sampling_rate = 100  # Hz

        # Compute the 2D positions
        positions = compute_position(data, sampling_rate)

        # Detect hard braking events
        g_force_threshold = 0.42  # Threshold in g
        deceleration_threshold = g_force_threshold * 9.81 / 3.6  # Convert g to m/s^2
        duration_threshold = 0.5  # Minimum duration of a hard braking event in seconds
        
        hard_braking_events = detect_hard_braking_events(data, deceleration_threshold, duration_threshold)
        
        # Plot the 2D route with speed heatmap and hard braking events
        plot_route_with_heatmap(positions, data['Speed'].values, hard_braking_events, participant_id, condition, output_dir)
    else:
        print("Data length is insufficient for analysis.")

def process_all_participants_data(data_logs_dir):
    """Process all participant data in the data_logs directory."""
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
                        process_single_file(csv_file, participant_id, condition, condition_path)

if __name__ == "__main__":
    data_logs_dir = os.path.join(os.path.dirname(__file__), '../data_logs')
    process_all_participants_data(data_logs_dir)
