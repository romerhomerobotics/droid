import h5py
import numpy as np
import cv2
import pyrealsense2 as rs
import os
import shutil
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# In droid_to_raw.py

# In droid_to_raw.py

def bridge_to_raw(droid_dir, output_episode_dir):
    droid_dir = os.path.normpath(droid_dir)
    os.makedirs(output_episode_dir, exist_ok=True)
    
    h5_path = os.path.join(droid_dir, "trajectory.h5")
    bag_dir = os.path.join(droid_dir, "recordings/Recordings")
    
    print(f"Processing: {droid_dir}")

    # --- 1. Process H5 Data (Robot State) ---
    with h5py.File(h5_path, 'r') as f:
        # Timestamps
        t_ms = f['observation']['timestamp']['control']['step_end'][:]
        t_rel = (t_ms / 1000.0).astype(np.float32)
        
        # EEF Poses
        cart_euler = f['action']['cartesian_position'][:]
        xyz = cart_euler[:, :3]
        euler_angles = cart_euler[:, 3:6]
        rot_objects = R.from_euler('xyz', euler_angles)
        quats = rot_objects.as_quat()
        eef_poses = np.hstack([t_rel[:, None], xyz, quats]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "eef_poses.npy"), eef_poses)

        # Gripper
        g_pos = f['action']['gripper_position'][:]
        g_pos_norm = g_pos / 0.068 
        gripper_data = np.hstack([t_rel[:, None], g_pos_norm[:, None]]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "gripper_positions.npy"), gripper_data)

        # Joint States
        j_pos = f['observation']['robot_state']['joint_positions'][:]
        joint_states = np.hstack([t_rel[:, None], j_pos]).astype(np.float32)
        np.save(os.path.join(output_episode_dir, "joint_states.npy"), joint_states)

    # --- 2. Process Cameras (Bags) ---
    bag_files = sorted([f for f in os.listdir(bag_dir) if f.endswith('.bag')])
    
    # Update this mapping if your camera serial numbers differ!
    # 0 = Hand (Wrist), 1 = Tripod (Ext1)
    cam_mapping = {
        0: {"role": "wrist", "ts_src": "022422070872_rgb_timestamps.npy"},
        1: {"role": "ext1", "ts_src": "135622077246_rgb_timestamps.npy"}
    }
    
    for i, bag in enumerate(bag_files):
        if i not in cam_mapping: continue
        
        role = cam_mapping[i]["role"]
        ts_src_name = cam_mapping[i]["ts_src"]
        bag_path = os.path.join(bag_dir, bag)
        src_ts_path = os.path.join(bag_dir, ts_src_name)

        print(f"[{role}] Processing bag: {bag}")
        
        # Setup Realsense Pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
        
        try:
            profile = pipeline.start(config)
            playback = rs.playback(profile.get_device())
            playback.set_real_time(False) # Fast processing
        except Exception as e:
            print(f"[{role}] ❌ Failed to open bag: {e}")
            continue

        rgb_frames = []
        depth_frames = []
        recovered_timestamps = []

        # --- Extraction Loop ---
        try:
            while True:
                # Wait for frames (increased timeout for safe reading)
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                
                # Get Color
                color_frame = frames.get_color_frame()
                if not color_frame:
                    break # End of bag

                # Get Depth
                depth_frame = frames.get_depth_frame()

                # Store Data
                rgb_frames.append(np.asanyarray(color_frame.get_data()))
                if depth_frame:
                    depth_frames.append(np.asanyarray(depth_frame.get_data()))

                # RECOVERY: If .npy is missing, read timestamp from frame metadata
                if not os.path.exists(src_ts_path):
                    # Backend timestamp is system time in ms
                    ts = color_frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
                    recovered_timestamps.append(ts)

        except RuntimeError:
            print(f"[{role}] End of stream or frame drop.")
        finally:
            pipeline.stop()

        # --- Save Timestamps ---
        final_ts_sec = None
        
        # Case A: Load pre-saved timestamps (Preferred)
        if os.path.exists(src_ts_path):
            print(f"[{role}] Loading existing timestamps from .npy")
            ts_ms = np.load(src_ts_path)
            final_ts_sec = (ts_ms / 1000.0).astype(np.float32)
            
        # Case B: Recovered from bag (Fallback)
        elif len(recovered_timestamps) > 0:
            print(f"[{role}] ⚠️ Recovered timestamps from BAG metadata (approximate sync)")
            rec_ts = np.array(recovered_timestamps, dtype=np.float64)
            # Normalize to start at 0.0s (Best guess since t0 is lost)
            rec_ts = rec_ts - rec_ts[0]
            final_ts_sec = (rec_ts / 1000.0).astype(np.float32)

        if final_ts_sec is not None:
            # Ensure length matches frames
            min_len = min(len(final_ts_sec), len(rgb_frames))
            final_ts_sec = final_ts_sec[:min_len]
            rgb_frames = rgb_frames[:min_len]
            if depth_frames: depth_frames = depth_frames[:min_len]

            np.save(os.path.join(output_episode_dir, f"{role}_color_timestamps.npy"), final_ts_sec)
            np.save(os.path.join(output_episode_dir, f"{role}_depth_timestamps.npy"), final_ts_sec)
        else:
            print(f"[{role}] ❌ CRITICAL: No timestamps found or recovered.")

        # --- Save Videos ---
        if len(rgb_frames) > 0:
            h, w = rgb_frames[0].shape[:2]
            # RGB
            out_v = os.path.join(output_episode_dir, f"{role}_color.mp4")
            vw = cv2.VideoWriter(out_v, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for frame in rgb_frames:
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"[{role}] Saved RGB video ({len(rgb_frames)} frames)")

            # Depth
            if len(depth_frames) > 0:
                out_d = os.path.join(output_episode_dir, f"{role}_depth.mp4")
                vw = cv2.VideoWriter(out_d, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                scale = 255.0 / 4000.0
                for d_map in depth_frames:
                    d_clipped = np.clip(d_map, 0, 4000).astype(np.float32)
                    d8 = (d_clipped * scale).astype(np.uint8)
                    vw.write(cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR))
                vw.release()
                print(f"[{role}] Saved Depth video")
        else:
            print(f"[{role}] ❌ No frames extracted!")

    print(f"[*] Success: Processed {droid_dir}")

if __name__ == "__main__":
    bridge_to_raw(
        "/home/robot/droid/data/success/2026-01-21/Wed_Jan_21_08:07:25_2026/", 
        "/home/robot/vla_data/raw/103"
    )