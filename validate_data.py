import os
import numpy as np
import cv2
import sys

# --- CONFIGURATION ---
# Path to the specific episode folder you want to validate
EPISODE_DIR = "/home/robot/vla_data/raw/106" 
# ---------------------

def analyze_timestamps(name, file_path, column_idx=None):
    """
    Loads an .npy file, prints shape, duration, start/end, and frequency.
    Returns the array if successful, None otherwise.
    """
    if not os.path.exists(file_path):
        print(f"❌ MISSING: {name} ({file_path})")
        return None
    
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"❌ ERROR LOADING {name}: {e}")
        return None

    # If it's 2D (like eef_poses), extract the timestamp column (usually col 0)
    # If it's 1D (like camera timestamps), use it directly
    if data.ndim > 1:
        if column_idx is None: column_idx = 0
        timestamps = data[:, column_idx]
    else:
        timestamps = data

    count = len(timestamps)
    if count < 2:
        print(f"⚠️ {name}: Not enough data points ({count})")
        return data

    t_start = timestamps[0]
    t_end = timestamps[-1]
    duration = t_end - t_start
    
    # Calculate Frequency
    diffs = np.diff(timestamps)
    if len(diffs) > 0:
        avg_dt = np.mean(diffs)
        min_dt = np.min(diffs)
        max_dt = np.max(diffs)
        freq = 1.0 / avg_dt if avg_dt > 0 else 0.0
    else:
        avg_dt, min_dt, max_dt, freq = 0, 0, 0, 0

    print(f"✅ {name}:")
    print(f"   - Shape: {data.shape}")
    print(f"   - Time Range: {t_start:.4f}s -> {t_end:.4f}s (Duration: {duration:.4f}s)")
    print(f"   - Frequency:  ~{freq:.2f} Hz (Avg dt: {avg_dt*1000:.2f}ms)")
    print(f"   - Jitter:     Min dt: {min_dt*1000:.2f}ms | Max dt: {max_dt*1000:.2f}ms")
    
    return data

def find_nearest_idx(array, value):
    """Finds index of element in sorted array closest to value."""
    if array is None or len(array) == 0:
        return 0
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx - 1
    return idx

def main():
    print(f"==================================================")
    print(f"   VALIDATING EPISODE: {EPISODE_DIR}")
    print(f"==================================================\n")

    if not os.path.isdir(EPISODE_DIR):
        print(f"❌ Error: Directory not found: {EPISODE_DIR}")
        return

    # --- 1. Print Saved Files ---
    print("[1] FILE LISTING")
    files = sorted(os.listdir(EPISODE_DIR))
    for f in files:
        size_mb = os.path.getsize(os.path.join(EPISODE_DIR, f)) / (1024 * 1024)
        print(f" - {f:<30} ({size_mb:.2f} MB)")
    print("-" * 50)

    # --- 2. Check Action Timestamps ---
    print("\n[2] ACTION DATA ANALYSIS")
    eef_poses = analyze_timestamps("eef_poses.npy", os.path.join(EPISODE_DIR, "eef_poses.npy"))
    analyze_timestamps("gripper_positions.npy", os.path.join(EPISODE_DIR, "gripper_positions.npy"))
    analyze_timestamps("joint_states.npy", os.path.join(EPISODE_DIR, "joint_states.npy"))
    print("-" * 50)

    # --- 3. Check Camera Timestamps ---
    print("\n[3] CAMERA TIMESTAMP ANALYSIS")
    wrist_ts = analyze_timestamps("wrist_color_timestamps.npy", os.path.join(EPISODE_DIR, "wrist_color_timestamps.npy"))
    ext1_ts = analyze_timestamps("ext1_color_timestamps.npy", os.path.join(EPISODE_DIR, "ext1_color_timestamps.npy"))
    
    # Check depth timestamps existence just to be safe
    analyze_timestamps("wrist_depth_timestamps.npy", os.path.join(EPISODE_DIR, "wrist_depth_timestamps.npy"))
    analyze_timestamps("ext1_depth_timestamps.npy", os.path.join(EPISODE_DIR, "ext1_depth_timestamps.npy"))
    print("-" * 50)

    # --- 4. Alignment Check ---
    if eef_poses is not None and wrist_ts is not None and len(wrist_ts) > 0 and len(eef_poses) > 0:
        print("\n[4] ALIGNMENT CHECK")
        print(f" - EEF Start:   {eef_poses[0,0]:.4f}s")
        print(f" - Wrist Start: {wrist_ts[0]:.4f}s")
        print(f" - Offset:      {abs(eef_poses[0,0] - wrist_ts[0]):.4f}s")
        if abs(eef_poses[0,0] - wrist_ts[0]) > 0.1:
            print("⚠️ WARNING: Large offset between camera and robot data start times!")
        else:
            print("✅ Alignment looks good.")
    else:
        print("\n[4] ALIGNMENT CHECK SKIPPED (Missing data)")
    print("-" * 50)

    # --- 5. Generate Validation Video ---
    print("\n[5] GENERATING VALIDATION VIDEO")
    
    wrist_vid_path = os.path.join(EPISODE_DIR, "wrist_color.mp4")
    ext1_vid_path = os.path.join(EPISODE_DIR, "ext1_color.mp4")
    output_vid_path = os.path.join(EPISODE_DIR, "validation_viz.mp4")

    # Graceful exit if missing videos
    has_wrist = os.path.exists(wrist_vid_path)
    has_ext1 = os.path.exists(ext1_vid_path)

    if not has_wrist and not has_ext1:
        print("❌ Both video files missing. Skipping video generation.")
        return

    # Open Video Captures
    cap_w = cv2.VideoCapture(wrist_vid_path) if has_wrist else None
    cap_e = cv2.VideoCapture(ext1_vid_path) if has_ext1 else None

    # Read first frames
    ret_w, frame_w = cap_w.read() if cap_w else (False, None)
    ret_e, frame_e = cap_e.read() if cap_e else (False, None)

    # Determine Dimensions
    if ret_w:
        H, W = frame_w.shape[:2]
    elif ret_e:
        H, W = frame_e.shape[:2]
    else:
        print("❌ Could not read frames from video files.")
        return

    # Handle missing stream by creating black frame
    if not ret_w: frame_w = np.zeros((H, W, 3), dtype=np.uint8)
    if not ret_e: frame_e = np.zeros((H, W, 3), dtype=np.uint8)

    # Resize ext1 to match wrist height for clean concatenation
    if ret_e: frame_e = cv2.resize(frame_e, (W, H))

    # Canvas Setup
    HEADER_H = 120
    canvas_w = W * 2
    canvas_h = H + HEADER_H
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid_path, fourcc, 30.0, (canvas_w, canvas_h))

    # Reset video pointers
    if cap_w: cap_w.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if cap_e: cap_e.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    
    # Safely get lengths
    len_w = len(wrist_ts) if wrist_ts is not None else 0
    len_e = len(ext1_ts) if ext1_ts is not None else 0
    
    # We want to run for at least as long as we have video frames
    frames_w = int(cap_w.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_w else 0
    frames_e = int(cap_e.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_e else 0
    
    max_frames = max(frames_w, frames_e)
    if max_frames == 0: max_frames = max(len_w, len_e) # Fallback to TS length
    
    print(f"Processing {max_frames} frames...")

    while frame_idx < max_frames:
        # Read next frames
        if cap_w:
            ret_w, fw = cap_w.read()
            if ret_w: frame_w = fw
        if cap_e:
            ret_e, fe = cap_e.read()
            if ret_e: frame_e = cv2.resize(fe, (W, H))

        # Get Timestamps (handle indices out of bounds)
        t_wrist = wrist_ts[frame_idx] if frame_idx < len_w else -1.0
        t_ext1 = ext1_ts[frame_idx] if frame_idx < len_e else -1.0

        # Get EEF Pose (Nearest Neighbor)
        eef_str = "EEF: N/A"
        if eef_poses is not None:
            # Use valid timestamp (prefer wrist, then ext1)
            t_query = t_wrist if t_wrist >= 0 else t_ext1
            if t_query >= 0:
                eef_idx = find_nearest_idx(eef_poses[:, 0], t_query)
                if 0 <= eef_idx < len(eef_poses):
                    p = eef_poses[eef_idx]
                    eef_str = (f"EEF (t={p[0]:.3f}s): "
                               f"Pos=[{p[1]:.3f}, {p[2]:.3f}, {p[3]:.3f}] "
                               f"Rot=[{p[4]:.2f}, {p[5]:.2f}, {p[6]:.2f}, {p[7]:.2f}]")

        # Create Canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Draw Header
        cv2.rectangle(canvas, (0, 0), (canvas_w, HEADER_H), (30, 30, 30), -1)
        cv2.putText(canvas, eef_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Frame: {frame_idx}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Place Images
        canvas[HEADER_H:, 0:W] = frame_w
        canvas[HEADER_H:, W:2*W] = frame_e

        # Overlay Timestamps
        color_w = (0, 255, 0) if t_wrist >= 0 else (0, 0, 255)
        color_e = (0, 255, 0) if t_ext1 >= 0 else (0, 0, 255)
        
        cv2.putText(canvas, f"WRIST: {t_wrist:.4f}s", (20, HEADER_H + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_w, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"EXT1: {t_ext1:.4f}s", (W + 20, HEADER_H + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_e, 2, cv2.LINE_AA)

        out.write(canvas)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f" -> {frame_idx}/{max_frames}", end='\r')

    if cap_w: cap_w.release()
    if cap_e: cap_e.release()
    out.release()
    print(f"\n✅ Video saved: {output_vid_path}")

if __name__ == "__main__":
    main()