import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import math
from geometry_msgs.msg import Quaternion # Used for converting Odometry Quaternion to Euler (optional)

# --- ROS 2 BAG IMPORTS ---
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from nav_msgs.msg import Odometry

# --- PART 1: ROS 2 BAG DATA EXTRACTION (Odometry: Position) ---
def extract_odom_data(bag_path, odom_topic='/dlio/odom_node/odom'):
    """
    Reads a ROS 2 bag and extracts Odometry data from a specified topic.
    """
    print(f"Starting extraction of Odometry data from: {bag_path}/ on topic: {odom_topic}")

    # List to store the extracted data
    odom_records = []
    
    try:
        # 1. Initialize Reader and Open Bag
        reader = rosbag2_py.SequentialReader()
        
        # NOTE: ROS 2 bags recorded since ROS Humble often use the 'mcap' storage ID.
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
        
        # 2. Filter Topics and Get Message Type
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        if odom_topic not in type_map:
            raise ValueError(f"Topic {odom_topic} not found in the bag file.")
            
        message_type = get_message(type_map[odom_topic])
        
        # 3. Read and Deserialize Messages
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            
            if topic == odom_topic:
                # Deserialize the binary message data into a ROS 2 Python object
                msg = deserialize_message(data, message_type)
                
                # Convert nanoseconds timestamp to seconds
                time_sec = timestamp_ns / 1e9 
                
                # Extract Position
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation

                odom_records.append({
                    'time_sec': time_sec,
                    'pos_x': position.x,
                    'pos_y': position.y,
                    'pos_z': position.z,
                    'quat_w': orientation.w,
                    'quat_x': orientation.x,
                    'quat_y': orientation.y,
                    'quat_z': orientation.z,
                })

        df_odom = pd.DataFrame(odom_records)
        print(f"Successfully extracted {len(df_odom)} odometry points.")
        return df_odom

    except ImportError:
        print("\nERROR: 'rosbag2_py' and/or ROS 2 message types are not available.")
        print("Did you remember to 'source /opt/ros/<distro>/setup.bash' before running this script?")
        return pd.DataFrame()
    except Exception as e:
        print(f"\nAn error occurred during bag reading: {e}")
        return pd.DataFrame()


# --- PART 2: LOG FILE DATA EXTRACTION (Performance Metrics) ---
def extract_performance_data(log_file_path):
    """
    Parses the dlio_odom_node terminal log file to extract performance metrics
    using regular expressions (Regex). (No change from original logic)
    """
    print(f"Parsing performance metrics from: {log_file_path}")

    # Dictionary to hold the extracted data lists
    data = {
        'elapsed_time': [], 'comp_time_ms': [], 'cores_utilized': [],
        'cpu_load_pct': [], 'ram_mb': []
    }

    # Regex patterns to capture the key values (assuming consistent formatting)
    patterns = {
        'elapsed_time': r'Elapsed Time:\s+(\d+\.\d+)\s+seconds',
        'comp_time_ms': r'Computation Time ::\s+(\d+\.\d+)\s+ms',
        'cores_utilized': r'Cores Utilized\s+::\s+(\d+\.\d+)\s+cores',
        'cpu_load_pct': r'CPU Load\s+::\s+(\d+\.\d+)\s+%',
        'ram_mb': r'RAM Allocation\s+::\s+(\d+\.\d+)\s+MB'
    }

    if not os.path.exists(log_file_path):
        print(f"Error: Log file '{log_file_path}' not found.")
        return pd.DataFrame()
        
    with open(log_file_path, 'r') as f:
        log_content = f.read()

    # Iterate through each performance metric
    for metric, pattern in patterns.items():
        # Find all occurrences of the pattern in the log content
        matches = re.findall(pattern, log_content)
        data[metric] = [float(match) for match in matches]

    # Ensure all lists have the same length (sanity check)
    if not all(len(data[list(data.keys())[0]]) == len(v) for v in data.values()):
        print("Warning: Performance data columns do not have equal length. Check log parsing logic.")

    df_perf = pd.DataFrame({k: v for k, v in data.items() if v})
    print(f"Extracted {len(df_perf)} performance snapshots.")
    return df_perf

# --- PART 3: VISUALIZATION ---
def visualize_data(df_odom, df_perf):
    """
    Generates the required plots for odometry and performance data. (Logic unchanged)
    """
    plt.style.use('ggplot')
    
    # Use the Elapsed Time from the performance log for all x-axes for alignment
    # If the odom data is denser, you'll need to interpolate perf data to align times.
    
    # 3D Position Plot
    fig = plt.figure(figsize=(12, 10))
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(df_odom['pos_x'], df_odom['pos_y'], df_odom['pos_z'], label='DLIO Trajectory')
    ax1.set_title('3D Trajectory (X, Y, Z)')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.legend()
    
    # Performance Plot 1: Timing
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df_perf['elapsed_time'], df_perf['comp_time_ms'], label='Comp. Time', color='red')
    ax2.set_title('Computation Time vs. Elapsed Time')
    ax2.set_xlabel('Elapsed Time (s)')
    ax2.set_ylabel('Comp. Time (ms)')
    ax2.legend()
    
    # Performance Plot 2: CPU/Cores
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(df_perf['elapsed_time'], df_perf['cpu_load_pct'], label='CPU Load (%)', color='blue')
    # Note: We plot Cores Utilized on the same axis for comparison, scaled by 10 for visibility.
    ax3.plot(df_perf['elapsed_time'], df_perf['cores_utilized'] * 10, label='Cores Utilized (Scaled x10)', color='green', linestyle='--')
    ax3.set_title('CPU Load and Core Utilization Over Time')
    ax3.set_xlabel('Elapsed Time (s)')
    ax3.set_ylabel('CPU Load (%) / Cores (Scaled)')
    ax3.legend()
    
    # Performance Plot 3: RAM
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(df_perf['elapsed_time'], df_perf['ram_mb'], label='RAM Allocation', color='purple')
    ax4.set_title('RAM Allocation Over Time')
    ax4.set_xlabel('Elapsed Time (s)')
    ax4.set_ylabel('RAM (MB)')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- Configuration ---
    # The directory created by ros2 bag record
    BAG_DIR = 'dlio_analysis_bag' 
    # The text file created by shell redirection (e.g., > dlio_output.txt 2>&1)
    LOG_FILE = 'dlio_output.txt' 
    
    # 1. Extract Odometry Data (Uses real rosbag2_py implementation now)
    odom_df = extract_odom_data(BAG_DIR)
    
    # 2. Extract Performance Data
    perf_df = extract_performance_data(LOG_FILE)
    
    # 3. Visualization
    if not odom_df.empty and not perf_df.empty:
        visualize_data(odom_df, perf_df)
    else:
        print("\nCould not visualize data. Check the error messages above for bag file or log file issues.")
