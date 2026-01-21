import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import traceback
from copy import deepcopy
from droid.misc.time import time_ms

logger = logging.getLogger(__name__)

# Helper mapping for resize functions
resize_func_map = {"cv2": cv2.resize, None: None}

def gather_realsense_cameras():
    """
    Detects all connected RealSense devices and returns a list of RealSenseCamera objects.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    all_cameras = []
    for dev in devices:
        cam = RealSenseCamera(dev)
        all_cameras.append(cam)
    return all_cameras

class RealSenseCamera:
    def __init__(self, device):
        # --- Save Parameters ---
        self.serial_number = device.get_info(rs.camera_info.serial_number)
        self.name = device.get_info(rs.camera_info.name)
        print(f"Opening RealSense: {self.name} (S/N: {self.serial_number})")

        self.pipeline = None
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        
        # State Tracking
        self.current_mode = "disabled"
        self._intrinsics = {}
        self.profile = None
        
        # --- NEW: Pipeline Synchronization Fields ---
        self.t0 = None
        self.recorded_timestamps = []
        self.bag_filepath = None
        
        # (Default flags and params remain unchanged)
        self.image = True
        self.depth = True
        self.skip_reading = False
        self.fps = 30
        self.default_hw_res = (1280, 720)
        self.align = rs.align(rs.stream.color)
        self.resize_func = None
        self.resizer_resolution = (0, 0)

    # ... (set_reading_parameters, set_trajectory_mode, set_calibration_mode remain unchanged) ...

    def start_recording(self, filepath, t0=None):
        """
        Starts recording a .bag file and initializes timestamp caching.
        """
        if filepath.endswith('.svo'):
            filepath = filepath.replace('.svo', '.bag')
        
        # Use provided t0 or default to now
        if t0 is None:
            raise ValueError("t0 must be provided to start_recording for timestamp alignment.")
        self.t0 = t0
        self.recorded_timestamps = []
        self.bag_filepath = filepath
        
        print(f"Starting recording for {self.serial_number} -> {filepath}")

        if self.pipeline:
            self.stop_recording()
        
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        self.config.enable_record_to_file(filepath)
        
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

    def stop_recording(self):
        """
        Closes the .bag file and saves the _rgb_timestamps.npy file for couple.py.
        """
        # Save timestamps before destroying the pipeline
        if self.bag_filepath and len(self.recorded_timestamps) > 0:
            ts_path = self.bag_filepath.replace(".bag", "_rgb_timestamps.npy")
            # Save as float32 relative seconds to match FineTuningDataCollector
            np.save(ts_path, np.array(self.recorded_timestamps, dtype=np.int32))
            print(f"[*] Saved {len(self.recorded_timestamps)} timestamps to {ts_path}")

        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            
        del self.profile
        del self.pipeline
        del self.config
        
        self.profile = None
        self.pipeline = None
        self.config = None
        self.bag_filepath = None
        self.current_mode = "disabled"

    def read_camera(self):
        """
        Reads frames and caches timestamps relative to t0.
        """
        if self.skip_reading or self.current_mode == "disabled":
            return {}, {}
        
        if self.pipeline is None:
            return None

        # Standard DROID timestamping
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            
            # --- NEW: Capture System-Aligned Timestamp for your pipeline ---
            color_frame = frames.get_color_frame()
            if color_frame and self.t0 is not None:
                # Use backend_timestamp (absolute system time in ms)
                raw_ms = color_frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
                rel_s = np.int32(raw_ms - self.t0)
                self.recorded_timestamps.append(rel_s)

        except RuntimeError:
            print(f"Frame drop or timeout on {self.serial_number}")
            return None

        # (Existing alignment and processing logic)
        try:
            frames = self.align.process(frames)
        except Exception as e:
            logger.error(f"Error processing frames: {e}")
            return None
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        timestamp_dict[self.serial_number + "_read_end"] = time_ms()
        timestamp_dict[self.serial_number + "_frame_received"] = frames.get_timestamp() 

        data_dict = {"image": {self.serial_number: self._process_frame(color_frame)}}
        if self.depth:
            data_dict["depth"] = {self.serial_number: self._process_frame(depth_frame)}

        return data_dict, timestamp_dict

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Non-Permanent Values (Trajectory) #
        self.traj_image = image
        self.traj_resolution = resolution

        # Permanent Values #
        self.depth = depth
        self.resize_func = resize_func_map[resize_func]
        
        # Logic for Hardware vs Software resolution (Mirroring ZED approach)
        if self.resize_func is None:
            self.hw_resolution = resolution if resolution != (0, 0) else self.default_hw_res
            self.resizer_resolution = (0, 0)
        else:
            self.hw_resolution = self.default_hw_res
            self.resizer_resolution = resolution

    def set_trajectory_mode(self):
        """
        Configures the camera for standard data collection.
        """
        # Set Parameters based on trajectory config
        self.image = self.traj_image
        self.skip_reading = not any([self.image, self.depth])

        # Configure and Launch Pipeline
        self._configure_pipeline()
        self.current_mode = "trajectory"

    def set_calibration_mode(self):
        """
        Sets the camera to a high-res mode for calibration.
        """
        if self.pipeline:
            self.disable_camera()

        # Calibration usually uses native high res
        self.hw_resolution = (1280, 720)
        self.image = True
        self.depth = True
        self.skip_reading = False
        
        self._configure_pipeline()
        self.current_mode = "calibration"

    def _configure_pipeline(self):
        if self.pipeline:
            self.disable_camera()

        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        
        w, h = self.hw_resolution
        print(f"Configuring RealSense {self.serial_number} for {w}x{h} @ {self.fps}fps")
        
        try:
            self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.fps)
            
            self.pipeline = rs.pipeline()
            self.profile = self.pipeline.start(self.config)
            
            # Cache Intrinsics immediately so get_intrinsics() is always safe
            stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream.get_intrinsics()
            self._intrinsics = {
                self.serial_number: {
                    "cameraMatrix": np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]),
                    "distCoeffs": np.array(intr.coeffs)
                }
            }
        except RuntimeError as e:
            print(f"Failed to start RealSense {self.serial_number} at {w}x{h}: {e}")
            raise

    def get_intrinsics(self):
        """Returns the cached intrinsics for the current stream profile."""
        return deepcopy(self._intrinsics)

    def _process_frame(self, frame):
        img = np.asanyarray(frame.get_data())
        if self.resizer_resolution == (0, 0) or self.resize_func is None:
            return img
        return self.resize_func(img, self.resizer_resolution)

    def disable_camera(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass 
        self.pipeline = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"