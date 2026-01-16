import cv2
import numpy as np
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

def test_pipeline():
    # Change this one value to update the entire system
    res = (640, 480) # use 640x480 for testing only, change to 1280x720 for real data collection deployment.

    camera_kwargs = {
        "hand_camera": {
            "image": True,
            "depth": True,
            "resolution": res,
            "resize_func": None, # Software resize if hardware bandwidth is an issue
        },
        "varied_camera": {
            "image": True,
            "depth": True,
            "resolution": res,
            "resize_func": None, # Software resize if hardware bandwidth is an issue
        }
    }
    print("Initializing MultiCameraWrapper...")
    wrapper = MultiCameraWrapper(camera_kwargs=camera_kwargs)
    
    print(f"Detected cameras: {list(wrapper.camera_dict.keys())}")

    try:
        while True:
            # DROID wrapper returns: (full_obs_dict, full_timestamp_dict)
            obs, timestamps = wrapper.read_cameras()

            if not obs or "image" not in obs:
                continue

            # In the DROID setup, images are stored in obs['image'][serial_number]
            for cam_id, image in obs["image"].items():
                # If depth was also captured
                window_name = f"DROID Stream: {cam_id}"
                
                # Check if depth is available for this camera
                if "depth" in obs and cam_id in obs["depth"]:
                    depth_raw = obs["depth"][cam_id]
                    depth_visual = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_raw, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    # Stack RGB and Depth for display
                    display_frame = np.hstack((image, depth_visual))
                else:
                    display_frame = image

                cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Closing cameras...")
        wrapper.disable_cameras()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pipeline()