import rosbag
import pandas as pd
import os
from datetime import datetime

def process_bag_files():
    rosbag_dir = os.path.join(os.path.dirname(__file__), '../rosbag_files')
    data_logs_dir = os.path.join(os.path.dirname(__file__), '../data_logs')

    # Ensure the output directory exists
    os.makedirs(data_logs_dir, exist_ok=True)

    # Process each bag file in the rosbag_files directory
    for bag_filename in os.listdir(rosbag_dir):
        if bag_filename.endswith('.bag'):
            bag_path = os.path.join(rosbag_dir, bag_filename)
            process_single_bag_file(bag_path, data_logs_dir)

def process_single_bag_file(bag_path, output_root_dir):
    # Extract participant ID and condition from the bag file name
    participant_id, condition = extract_participant_and_condition_from_filename(bag_path)

    # Create output directories based on participant ID and condition
    output_dir = os.path.join(output_root_dir, f'participant_{participant_id}', f'condition_{condition}')

    # Ensure the output directory for the specific participant and condition exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the bag file
    bag = rosbag.Bag(bag_path)

    # Gather topics and types information
    topics_info = {
        "Topic": [],
        "Type": [],
        "Message Count": []
    }
    for topic, type_info in bag.get_type_and_topic_info().topics.items():
        topics_info["Topic"].append(topic)
        topics_info["Type"].append(type_info.msg_type)
        topics_info["Message Count"].append(type_info.message_count)

    # Convert to DataFrame for better readability
    topics_df = pd.DataFrame(topics_info)

    # Save topics information to CSV
    topics_info_csv_path = os.path.join(output_dir, 'topics_info.csv')
    topics_df.to_csv(topics_info_csv_path, index=False)

    # Dictionary to hold dataframes for each topic
    topic_dataframes = {}

    # Extract all messages and sort them by topic
    for topic, msg, t in bag.read_messages():
        if topic not in topic_dataframes:
            topic_dataframes[topic] = {
                "Time": [],
                "Message": []
            }
        topic_dataframes[topic]["Time"].append(t.to_sec())
        topic_dataframes[topic]["Message"].append(str(msg))

    # Save each topic to its own CSV file
    for topic, data in topic_dataframes.items():
        topic_df = pd.DataFrame(data)
        # Replace slashes in topic name to create a valid filename
        sanitized_topic = topic.replace('/', '_')
        csv_path = os.path.join(output_dir, f'{sanitized_topic}.csv')
        topic_df.to_csv(csv_path, index=False)

    # Display file paths of the created CSV files
    print("Topics information saved to:", topics_info_csv_path)
    for topic in topic_dataframes:
        sanitized_topic = topic.replace('/', '_')
        csv_path = os.path.join(output_dir, f'{sanitized_topic}.csv')
        print(f"Messages for topic '{topic}' saved to:", csv_path)

    # Close the bag file
    bag.close()

def extract_participant_and_condition_from_filename(bag_path):
    """Extract participant ID and condition from the filename, assuming the format 'operator_<participant_id>_<condition>_YYYY-MM-DD-HH-MM-SS.bag'."""
    filename = os.path.basename(bag_path)
    parts = filename.split('_')
    participant_id = parts[1]  # Extract participant ID
    condition = parts[2]       # Extract condition (A, B, C)
    return participant_id, condition

def main():
    process_bag_files()

if __name__ == "__main__":
    main()
