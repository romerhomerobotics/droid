import h5py
import numpy as np
import os
import shutil
import argparse
from scipy.spatial.transform import Rotation as R

def bridge_to_raw(droid_dir, output_episode_dir):
    droid_dir = os.path.normpath(droid_dir)
    os.makedirs(output_episode_dir, exist_ok=True)
    
    h5_path = os.path.join(droid_dir, "trajectory.h5")
    
    # Locate recordings directory (Handle potential nesting)
    rec_dir = os.path.join(droid_dir, "recordings/Recordings")
    if not os.path.exists(rec_dir):
        rec_dir = os.path.join(droid_dir, "recordings")
    
    print(f"Processing: {droid_dir}")

    # 1. Process H5 (Robot State)
    with h5py.File(h5_path, 'r') as f:
        # Timestamps (ms -> sec float32)
        t_ms = f['observation']['timestamp']['control']['step_end'][:]
        t_rel = (t_ms / 1000.0).astype(np.float32)
        
        # --- EEF Poses (With 45deg Correction) ---
        cart = f['action']['cartesian_position'][:]
        xyz = cart[:, :3]
        
        # Convert DROID Euler (XYZ) to Rotation Object
        rot_droid = R.from_euler('xyz', cart[:, 3:6])
        
        # Create Correction: 45 degrees around Z-axis
        correction = R.from_euler('z', 45, degrees=True)
        
        # Apply correction (Global Frame: Correction * Current)
        rot_corrected = correction * rot_droid
        quat = rot_corrected.as_quat() # [x, y, z, w]
        
        # Save [t, x, y, z, qx, qy, qz, qw]
        eef_poses = np.hstack([t_rel[:,None], xyz, quat]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "eef_poses.npy"), eef_poses)

        # Gripper (Normalize 0.068 -> 1.0)
        g_pos = f['action']['gripper_position'][:] 
        g_pos_norm = g_pos / 0.068
        
        gripper_data = np.hstack([t_rel[:,None], g_pos_norm[:,None]]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "gripper_positions.npy"), gripper_data)

        # Joints
        j_pos = f['observation']['robot_state']['joint_positions'][:]
        joint_states = np.hstack([t_rel[:,None], j_pos]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "joint_states.npy"), joint_states)

    # --- 2. Generate events.npz ---
    # Required to match data_collector.py format exactly
    events = {
        "reached_object": np.empty((0,), dtype=np.float32),
        "gripped_object": np.empty((0,), dtype=np.float32),
        "placed_object": np.empty((0,), dtype=np.float32),
        "dropped_object": np.empty((0,), dtype=np.float32)
    }
    np.savez(os.path.join(output_episode_dir, "events.npz"), **events)

    # --- 3. Process Cameras (Rename & Move) ---
    cam_mapping = {
        "022422070872": "ext1",
        "135622077246": "wrist"
    }

    for serial, role in cam_mapping.items():
        # Source files created by realsense_camera.py
        src_color = os.path.join(rec_dir, f"{serial}_color.mp4")
        src_depth = os.path.join(rec_dir, f"{serial}_depth.mp4")
        src_ts    = os.path.join(rec_dir, f"{serial}_timestamps.npy")
        
        if not os.path.exists(src_color):
            print(f"[{role}] Missing video: {src_color}")
            continue

        # Copy Video
        shutil.copy(src_color, os.path.join(output_episode_dir, f"{role}_color.mp4"))
        if os.path.exists(src_depth):
            shutil.copy(src_depth, os.path.join(output_episode_dir, f"{role}_depth.mp4"))
        
        # Copy Timestamps (used for both RGB and Depth)
        if os.path.exists(src_ts):
            shutil.copy(src_ts, os.path.join(output_episode_dir, f"{role}_color_timestamps.npy"))
            shutil.copy(src_ts, os.path.join(output_episode_dir, f"{role}_depth_timestamps.npy"))
            print(f"[{role}] Copied video and timestamps.")

    print(f"[*] Success: Raw data generated in {output_episode_dir}")
    return True

def batch_process_droid(droid_dir, output_dir):
    """Gathers all droid-teleop sample folders under droid_dir and converts them into numbered outputs saved under output_dir. """
    # Find all subdirectories that contain a trajectory.h5 file
    if not os.path.exists(droid_dir):
        print(f"Error: Source directory {droid_dir} does not exist.")
        return

    samples = sorted([
        os.path.join(droid_dir, d) for d in os.listdir(droid_dir)
        if os.path.isdir(os.path.join(droid_dir, d))
    ])
    
    print(f"Found {len(samples)} potential samples in {droid_dir}")
    
    count = 0
    for i, sample_path in enumerate(samples, start=1):
        # Create output path: /home/robot/vla_data/output_dir/i
        output_path = os.path.join(output_dir, str(i))
        
        try:
            success = bridge_to_raw(sample_path, output_path)
            if success:
                count += 1
        except Exception as e:
            print(f"Failed to process {sample_path}: {e}")
            
    print(f"\nBatch processing complete. Successfully converted {count}/{len(samples)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert DROID samples to Raw VLA format.")
    parser.add_argument("--droid_dir", type=str, required=True, 
                        help="Path to the directory containing DROID sample folders.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="The base output directory (e.g. /home/robot/vla_data/raw). Episodes will be saved as 1, 2, 3... inside.")

    args = parser.parse_args()
    
    batch_process_droid(args.droid_dir, args.output_dir)