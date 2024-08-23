import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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

def detect_hard_braking_events(data, deceleration_threshold, duration_threshold):
    """Detect hard braking events based on a given deceleration threshold and a minimum duration."""
    events = []
    event_start = None

    for i in range(1, len(data)):
        speed_change = data['LongitudinalSpeed'].iloc[i - 1] - data['LongitudinalSpeed'].iloc[i]
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
                        events.append((event_start, data['TimeSeconds'].iloc[i - 1]))
                    event_start = None

    # Handle case where the event continues till the end
    if event_start is not None and (data['TimeSeconds'].iloc[-1] - event_start) >= duration_threshold:
        events.append((event_start, data['TimeSeconds'].iloc[-1]))

    return events

def plot_hard_braking(data, events, participant_id, condition, output_dir, time_range=None):
    """Plot longitudinal speed with hard braking events highlighted."""
    plt.figure(figsize=(14, 8))

    # Plot the longitudinal speed
    speed_line, = plt.plot(data['TimeSeconds'], data['LongitudinalSpeed'], marker='o', color='#89CFF0', markersize=4, label='Longitudinal Speed')

    for start, end in events:
        plt.axvspan(start, end, color='#FFB6C1', alpha=0.3, label='Hard Braking Event' if start == events[0][0] else "")

    # Zoom in on the specified time range if provided
    if time_range:
        plt.xlim(time_range)

    # Add legend
    plt.legend(handles=[speed_line], loc='upper right')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Longitudinal Speed (m/s)')
    title = f'Hard Braking Events - Participant {participant_id}, Condition {condition}'
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    
    # Create the filename based on the plot title
    sanitized_title = re.sub(r'\W+', '_', title)  # Replace non-alphanumeric characters with underscores
    output_path = os.path.join(output_dir, f"{sanitized_title}.png")
    
    plt.savefig(output_path)
    plt.close()


def process_single_file(file_path, participant_id, condition, output_dir):
    """Process a single CSV file and return the total hard braking events and session length."""
    data = load_data(file_path)
    if data is None:
        return 0, "0:00"

    data['LongitudinalSpeed'] = data['Message'].apply(extract_longitudinal_speed)
    data = data.dropna(subset=['LongitudinalSpeed'])

    data['TotalSeconds'] = data['Message'].apply(extract_time)
    data['TimeSeconds'] = data['TotalSeconds'] - data['TotalSeconds'].min()  # Time in seconds from start

    if len(data['LongitudinalSpeed']) > 9:
        # Convert the threshold from g-force to km/h per second for the longitudinal speed in m/s
        g_force_threshold = 0.42  # Threshold in g
        deceleration_threshold = g_force_threshold * 9.81 / 3.6  # Convert g to m/s^2
        
        # Minimum duration to consider as a hard braking event (in seconds)
        duration_threshold = 0.5  # Example: Event must last at least 0.5 seconds
        
        # Detect hard braking events
        hard_braking_events = detect_hard_braking_events(data, deceleration_threshold, duration_threshold)
    
        # Calculate the total duration of the session in seconds
        total_duration = data['TimeSeconds'].max() - data['TimeSeconds'].min()

        # Convert total_duration to minutes and seconds format
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        session_length = f"{minutes}:{seconds:02d}"
        
        # Include a time range for plotting if needed
        time_range = (50, 150)  # Zoom in on time between 50 and 150 seconds

        # Save the plot as a PNG file with a title-based filename
        plot_hard_braking(data, hard_braking_events, participant_id, condition, output_dir, time_range=time_range)
        
        return len(hard_braking_events), session_length
    else:
        print("Data length is insufficient for analysis.")
        return 0, "0:00"


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
                        total_events, session_length = process_single_file(csv_file, participant_id, condition, condition_path)

                        # Save the results
                        results.append({
                            "Participant ID": participant_id,
                            "Condition": condition,
                            "Hard Braking Events": total_events,
                            "Session Length (min:sec)": session_length
                        })
    # Save the results to a main CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_logs_dir, 'hard_braking_summary.csv'), index=False)

if __name__ == "__main__":
    data_logs_dir = os.path.join(os.path.dirname(__file__), '../data_logs')
    process_all_participants_data(data_logs_dir)
