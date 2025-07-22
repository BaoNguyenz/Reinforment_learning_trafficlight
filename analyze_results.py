import xml.etree.ElementTree as ET
import numpy as np
import os

def analyze_tripinfo(file_path=r"E:\LET ME COOK\REL301m\dqn_traffic_project\sumo_files\tripinfo.xml"):
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please run evaluate.py or train.py to generate this file first.")
        return

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error: File '{file_path}' is empty or contains invalid XML.")
        return

    durations = []
    time_losses = []
    wait_times = []

    for trip in root.findall('tripinfo'):
        durations.append(float(trip.get('duration')))
        time_losses.append(float(trip.get('timeLoss')))
        wait_times.append(float(trip.get('waitingTime')))

    if len(durations) > 0:
        avg_duration = np.mean(durations)
        avg_time_loss = np.mean(time_losses)
        avg_wait_time = np.mean(wait_times)
        
        print("\n--- TRAFFIC PERFORMANCE ANALYSIS RESULTS ---")
        print(f"Total number of vehicles that completed their trips: {len(durations)}")
        print(f"Average travel time per vehicle: {avg_duration:.2f} seconds")
        print(f"Average time loss (due to congestion): {avg_time_loss:.2f} seconds")
        print(f"Average waiting time: {avg_wait_time:.2f} seconds")
        print("-------------------------------------------------")
    else:
        print("No vehicles completed their trips during the simulation.")
        print("This indicates a very severe traffic congestion.")

if __name__ == '__main__':
    analyze_tripinfo()
