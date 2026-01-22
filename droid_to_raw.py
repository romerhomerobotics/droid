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
    # Camera files are now saved here by realsense_camera.py
    # Note: verify if it is recording/Recordings or just recordings depending on your setup
    rec_dir = os.path.join(droid_dir, "recordings/Recordings") 
    
    print(f"Processing: {droid_dir}")

    # 1. Process H5 (Robot State)
    with h5py.File(h5_path, 'r') as f:
        t_ms = f['observation']['timestamp']['control']['step_end'][:]
        t_rel = (t_ms / 1000.0).astype(np.float32)
        
        # EEF
        cart = f['action']['cartesian_position'][:]
        xyz = cart[:, :3]
        quat = R.from_euler('xyz', cart[:, 3:6]).as_quat()
        np.save(os.path.join(output_episode_dir, "eef_poses.npy"), 
                np.hstack([t_rel[:,None], xyz, quat]).astype(np.float32))

        # Gripper
        g_pos = f['action']['gripper_position'][:] / 0.068
        np.save(os.path.join(output_episode_dir, "gripper_positions.npy"), 
                np.hstack([t_rel[:,None], g_pos[:,None]]).astype(np.float32))

        # Joints
        j_pos = f['observation']['robot_state']['joint_positions'][:]
        np.save(os.path.join(output_episode_dir, "joint_states.npy"), 
                np.hstack([t_rel[:,None], j_pos]).astype(np.float32))

    # 2. Process Cameras (Rename & Move)
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

#if __name__ == "__main__":
#    # Update paths as needed
#    bridge_to_raw(
#        "/home/robot/droid/data/success/2026-01-21/Wed_Jan_21_12:43:17_2026/", 
#        "/home/robot/vla_data/raw/106"
#    )

def batch_process_droid(droid_dir, output_dir):
    """Gathers all droid-teleop sample folders under droid_dir and converts them into numbered outputs saved under output_dir. """
    # Find all subdirectories that contain a trajectory.h5 file
    # Sorting ensures that the 1...n numbering is deterministic
    samples = sorted([
        os.path.join(droid_dir, d) for d in os.listdir(droid_dir)
        if os.path.isdir(os.path.join(droid_dir, d))
    ])
    
    print(f"Found {len(samples)} potential samples in {droid_dir}")
    
    count = 0
    for i, sample_path in enumerate(samples, start=1):
        # Create output path: /home/robot/vla_data/output_dir/i
        output_path = os.path.join("", output_dir, str(i))
        
        success = bridge_to_raw(sample_path, output_path)
        if success:
            count += 1
            
    print(f"\nBatch processing complete. Successfully converted {count}/{len(samples)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert DROID samples to Raw VLA format.")
    parser.add_argument("--droid_dir", type=str, required=True, 
                        help="Path to the directory containing DROID sample folders.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="The name to use for the output folder (e.g., 'success_2026_01_21').")

    args = parser.parse_args()
    
    batch_process_droid(args.droid_dir, args.output_dir)