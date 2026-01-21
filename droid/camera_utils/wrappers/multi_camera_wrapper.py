import os
import random
from collections import defaultdict

# --- CHANGED: Import RealSense instead of ZED ---
from droid.camera_utils.camera_readers.realsense_camera import gather_realsense_cameras
from droid.camera_utils.info import get_camera_type

class MultiCameraWrapper:
    def __init__(self, camera_kwargs={}):
        # Open Cameras #
        # --- CHANGED: Gather RealSense ---
        rs_cameras = gather_realsense_cameras()
        self.camera_dict = {cam.serial_number: cam for cam in rs_cameras}

        if len(self.camera_dict) == 0:
            print("WARNING: No RealSense cameras found.")

        # Set Correct Parameters #
        for cam_id in self.camera_dict.keys():
            cam_type = get_camera_type(cam_id)
            curr_cam_kwargs = camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

        # Launch Camera #
        self.set_trajectory_mode()

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def set_calibration_mode(self, cam_id):
        # If any camera is in calibration mode, close others
        # (RealSense bandwidth is high, safer to run one at a time for calib)
        close_all = any(
            [cam.current_mode == "calibration" for cam in self.camera_dict.values()]
        )

        if close_all:
            for curr_cam_id in self.camera_dict:
                if curr_cam_id != cam_id:
                    self.camera_dict[curr_cam_id].disable_camera()

        self.camera_dict[cam_id].set_calibration_mode()

    def set_trajectory_mode(self):
        # Close all if switching back from calibration
        close_all = any(
            [cam.current_mode == "calibration" for cam in self.camera_dict.values()]
        )

        if close_all:
            for cam in self.camera_dict.values():
                cam.disable_camera()

        # Put All Cameras In Trajectory Mode #
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath, t0=None):
        # --- CHANGED: Use 'Recordings' folder and .bag extension ---
        subdir = os.path.join(recording_folderpath, "Recordings")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
            
        for cam in self.camera_dict.values():
            # RealSense driver handles the pipeline restart internally
            filepath = os.path.join(subdir, cam.serial_number + ".bag")
            cam.start_recording(filepath, t0=t0)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def read_cameras(self):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if not self.camera_dict[cam_id].is_running():
                continue
            
            # Read Data
            result = self.camera_dict[cam_id].read_camera()
            if result is None: 
                continue
                
            data_dict, timestamp_dict = result

            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
            full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict

    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.disable_camera()