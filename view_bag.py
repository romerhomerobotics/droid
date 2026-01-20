import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Path to your successful recording
DIR = "/home/robot/droid/data/success/2026-01-20/Tue_Jan_20_12:29:42_2026/recordings/Recordings/"
BAG_FILES = sorted([f for f in os.listdir(DIR) if f.endswith('.bag')])

def play_all_cameras(bag_files):
    pipelines = []
    
    # Initialize a pipeline for every bag file found
    for bag in bag_files:
        path = os.path.join(DIR, bag)
        pipe = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, path)
        pipe.start(cfg)
        pipelines.append(pipe)
        print(f"Started pipeline for: {bag}")

    print("\nDisplaying all streams. Press 'q' to quit.")
    
    try:
        while True:
            frames_list = []
            for pipe in pipelines:
                frames = pipe.wait_for_frames(5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frames_list.append(np.asanyarray(color_frame.get_data()))

            if not frames_list:
                break

            # Create a horizontal stack of all camera feeds
            # If you have 3+ cameras, you can use np.vstack to create a grid
            combined_view = np.hstack(frames_list)
            
            # Resize for easier viewing if the combined image is too wide
            h, w = combined_view.shape[:2]
            scale = 0.5
            resized_view = cv2.resize(combined_view, (int(w * scale), int(h * scale)))
            
            cv2.imshow("Multi-Camera Inspection", resized_view)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"\nPlayback ended: {e}")
    finally:
        for pipe in pipelines:
            pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not BAG_FILES:
        print(f"No .bag files found in {DIR}")
    else:
        play_all_cameras(BAG_FILES)
