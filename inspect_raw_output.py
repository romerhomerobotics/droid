import numpy as np
import cv2
import os

# Update to the path where your droid_to_raw.py saved the data
RAW_DIR = "/home/robot/vla_data/raw/101"

def inspect_raw():
    print(f"--- Inspecting Raw Output in: {RAW_DIR} ---\n")

    # 1. Check eef_poses.npy (Expected: [t, x, y, z, qx, qy, qz, qw])
    pose_path = os.path.join(RAW_DIR, "eef_poses.npy")
    if os.path.exists(pose_path):
        poses = np.load(pose_path)
        print(f"[EEF Poses] Shape: {poses.shape} (Expected: N x 8)")
        print(f"  - First 5 Timestamps: {poses[:5, 0].round(4)}")
        print(f"  - t=0 check: {'SUCCESS' if poses[0, 0] == 0 else 'WARNING: t0 is not 0'}")
        
        # Verify Quaternion Norm (should be ~1.0)
        quat_norm = np.linalg.norm(poses[0, 4:8])
        print(f"  - Quat Norm Check: {quat_norm:.4f} (Expected: 1.0)")
    else:
        print("[ERROR] eef_poses.npy missing!")

    # 2. Check gripper_positions.npy
    grip_path = os.path.join(RAW_DIR, "gripper_positions.npy")
    if os.path.exists(grip_path):
        grip = np.load(grip_path)
        print(f"\n[Gripper] Shape: {grip.shape} (Expected: N x 2)")
        print(f"  - Range: {grip[:, 1].min():.3f} to {grip[:, 1].max():.3f}")
    else:
        print("[ERROR] gripper_positions.npy missing!")

    # 3. Check Videos & Timestamps Alignment
    for cam in ["hand", "front"]:
        v_path = os.path.join(RAW_DIR, f"{cam}_color.mp4")
        ts_path = os.path.join(RAW_DIR, f"{cam}_rgb_timestamps.npy")
        
        if os.path.exists(v_path) and os.path.exists(ts_path):
            ts = np.load(ts_path)
            cap = cv2.VideoCapture(v_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            print(f"\n[{cam.upper()} CAMERA]")
            print(f"  - Video Frames: {frame_count}")
            print(f"  - Timestamp Count: {len(ts)}")
            print(f"  - Sync Match: {'MATCH' if frame_count == len(ts) else 'MISMATCH'}")
            print(f"  - Start Time: {ts[0]:.4f}s")
        else:
            print(f"\n[WARNING] {cam} camera files partially missing.")

if __name__ == "__main__":
    inspect_raw()
