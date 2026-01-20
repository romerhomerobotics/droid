import h5py
import numpy as np
import cv2
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def bridge_to_raw(droid_dir, output_episode_dir):
    os.makedirs(output_episode_dir, exist_ok=True)
    h5_path = os.path.join(droid_dir, "trajectory.h5")
    bag_dir = os.path.join(droid_dir, "recordings/Recordings")
    
    with h5py.File(h5_path, 'r') as f:
        # 1. Establish master t0 from the robot's first step (DROID uses SECONDS)
        t_raw = f['observation']['timestamp']['control']['step_end'][:]
        t0_robot = t_raw[0]
        t_rel = (t_raw - t0_robot).astype(np.float32)

        # 2. Convert DROID Euler (XYZ) to Quaternions (XYZW) for couple.py
        cart_euler = f['action']['cartesian_position'][:]
        xyz = cart_euler[:, :3]
        euler_angles = cart_euler[:, 3:6]
        
        rot_objects = R.from_euler('xyz', euler_angles)
        quats = rot_objects.as_quat() # Returns [qx, qy, qz, qw]
        
        # Combine into [t, x, y, z, qx, qy, qz, qw]
        eef_poses = np.hstack([t_rel[:, None], xyz, quats]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "eef_poses.npy"), eef_poses)

        # 3. Gripper Data (Normalized g_pos)
        g_pos = f['action']['gripper_position'][:]
        gripper_data = np.hstack([t_rel[:, None], g_pos[:, None]]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "gripper_positions.npy"), gripper_data)

        # 4. Extract Videos and Hardware-Synced Timestamps
        bag_files = sorted([f for f in os.listdir(bag_dir) if f.endswith('.bag')])
        
        # Mapping to match your couple.py schema
        cam_mapping = {
            0: {"vid": "hand_color.mp4", "ts": "hand_rgb_timestamps.npy"},
            1: {"vid": "front_color.mp4", "ts": "front_rgb_timestamps.npy"}
        }
        
        for i, bag in enumerate(bag_files):
            if i not in cam_mapping: continue
            
            bag_path = os.path.join(bag_dir, bag)
            pipeline = rs.pipeline()
            config = rs.config()
            rs.config.enable_device_from_file(config, bag_path, False)
            
            profile = pipeline.start(config)
            playback = rs.playback(profile.get_device())
            playback.set_real_time(False)

            frames = []
            timestamps = []
            
            print(f"[*] Processing {bag} -> {cam_mapping[i]['vid']}")
            
            try:
                while True:
                    # Use a shorter timeout as we are in non-real-time mode
                    fs = pipeline.wait_for_frames(2000)
                    color = fs.get_color_frame()
                    if not color: break
                    
                    # --- TIMESTAMP ALIGNMENT LOGIC ---
                    # backend_timestamp is absolute system time
                    raw_ts = color.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
                    
                    # Detect if units are Microseconds (16 digits) or Milliseconds (13 digits)
                    # Unix time in seconds is ~1.7e9.
                    if raw_ts > 1e14:
                        ts_seconds = raw_ts / 1_000_000.0 # Micro to Sec
                    else:
                        ts_seconds = raw_ts / 1_000.0     # Milli to Sec
                    
                    # Shift relative to robot t0
                    rel_ts_s = ts_seconds - t0_robot
                    
                    timestamps.append(rel_ts_s)
                    img = np.asanyarray(color.get_data())
                    frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except RuntimeError:
                pass # End of bag file
            finally:
                pipeline.stop()

            # Save MP4
            if frames:
                h, w = frames[0].shape[:2]
                out_v = os.path.join(output_episode_dir, cam_mapping[i]["vid"])
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cv2.VideoWriter(out_v, fourcc, 30, (w, h))
                for frame in frames:
                    vw.write(frame)
                vw.release()
                
                # Save .npy timestamps (Relative Seconds)
                np.save(os.path.join(output_episode_dir, cam_mapping[i]["ts"]), 
                        np.array(timestamps, dtype=np.float32))

    print(f"[*] Success: Episode raw data generated in {output_episode_dir}")

if __name__ == "__main__":
    bridge_to_raw(
        "/home/robot/droid/data/success/2026-01-20/Tue_Jan_20_12:29:42_2026", 
        "/home/robot/vla_data/raw/100"
    )