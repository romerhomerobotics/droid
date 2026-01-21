import pyrealsense2 as rs
import numpy as np
import os

DIR = "/home/robot/droid/data/success/2026-01-20/Tue_Jan_20_14:23:01_2026/recordings/Recordings"
BAG_FILES = sorted([f for f in os.listdir(DIR) if f.endswith('.bag')])

def rigorous_validation(bag_path):
    if not os.path.exists(bag_path):
        print(f"File not found: {bag_path}")
        return

    # Calculate File Stats
    file_size_mb = os.path.getsize(bag_path) / (1024 * 1024)

    pipeline = rs.pipeline()
    config = rs.config()
    
    # FIX: Pass repeat as a positional argument to avoid TypeError
    rs.config.enable_device_from_file(config, bag_path, False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = rs.playback(device)
    playback.set_real_time(False) 
    
    hw_timestamps = []
    frame_numbers = []
    
    print(f"\n--- Rigorous Analysis: {os.path.basename(bag_path)} ---")
    
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            hw_timestamps.append(color_frame.get_timestamp())
            frame_numbers.append(color_frame.get_frame_number())
            
    except RuntimeError:
        pass # End of file
    finally:
        pipeline.stop()

    if len(hw_timestamps) < 2:
        print("  [ERROR] No frames captured from bag.")
        return

    # ANALYSIS
    hw_timestamps = np.array(hw_timestamps)
    frame_numbers = np.array(frame_numbers)
    intervals = np.diff(hw_timestamps)
    f_diffs = np.diff(frame_numbers)
    
    duration_sec = (hw_timestamps[-1] - hw_timestamps[0]) / 1000.0
    avg_int = np.mean(intervals)
    true_hz = 1000.0 / avg_int
    bitrate = file_size_mb / duration_sec if duration_sec > 0 else 0
    
    total_missed = np.sum(f_diffs - 1) if len(f_diffs) > 0 else 0
    duplicates = np.sum(f_diffs == 0)

    print(f"  - Duration: {duration_sec:.2f} seconds")
    print(f"  - File Size: {file_size_mb:.2f} MB")
    print(f"  - Write Throughput: {bitrate:.2f} MB/s")
    print(f"  - Sensor-Level Hz: {true_hz:.2f} Hz")
    print(f"  - Hardware Frame Drops: {total_missed} frames")
    print(f"  - Software Duplicates: {duplicates}")
    
    # Final Verdict
    if total_missed == 0 and duplicates == 0 and abs(true_hz - 30) < 1.5:
        print("  STATUS: [GOLDEN] Perfect capture.")
    elif total_missed == 0 and duplicates > 0:
        print("  STATUS: [OK-ISH] No data lost, but script double-saved frames (Disk/CPU lag).")
    else:
        print("  STATUS: [BAD] Hardware frames were dropped. Recording is unreliable.")

if __name__ == "__main__":
    if not BAG_FILES:
        print(f"No .bag files found in {DIR}")
    for bag in BAG_FILES:
        rigorous_validation(os.path.join(DIR, bag))
