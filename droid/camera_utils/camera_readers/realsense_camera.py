import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import traceback
from copy import deepcopy
from droid.misc.time import time_ms
import os

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
        self.recording_dir = None
        self.frame_buffer = []  # <--- Buffer for (timestamp, image)
        self.recording_active = False
        
        # (Default flags and params remain unchanged)
        self.image = True
        self.depth = True
        self.skip_reading = False
        self.fps = 30
        self.default_hw_res = (1280, 720)
        self.align = rs.align(rs.stream.color)
        self.resize_func = None
        self.resizer_resolution = (0, 0)

    def start_recording(self, recording_dir, t0=None):
            """
            Starts buffering frames to RAM.
            """
            if t0 is None:
                raise ValueError("t0 must be provided.")
            
            self.t0 = t0
            if recording_dir.endswith('.bag') or recording_dir.endswith('.svo'):
                self.recording_dir = os.path.dirname(recording_dir)
            else:
                self.recording_dir = recording_dir
            self.frame_buffer = [] # Clear previous buffer
            self.recording_active = True
            
            print(f"[{self.serial_number}] Buffering started (RAM).")
            
            # Ensure pipeline is running
            if not self.pipeline:
                self._configure_pipeline()
                self.current_mode = "trajectory"
        # -----------------------------------------------

    def read_camera(self):
        print(f"read_camera start: Reading from RealSense {self.serial_number}...")
        if self.skip_reading or self.current_mode == "disabled":
            return {}, {}
        
        if self.pipeline is None:
            return None

        t_start = time_ms() # Capture start time

        try:
            # Increased timeout slightly to ensure we don't drop frames unnecessarily 
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            
            color_frame = frames.get_color_frame()
            if color_frame:
                # FIX 1: Use .copy() so Python owns the data, not the RS circular buffer
                c_img = np.asanyarray(color_frame.get_data()).copy()
                
                if self.recording_active and self.t0 is not None:
                    rel_s = (t_start - self.t0) / 1000.0 
                    self.frame_buffer.append((rel_s, c_img))
            
            # FIX 2: Call time_ms() again for read_end, otherwise it's identical to start
            t_end = time_ms()
            
            timestamp_dict = {
                self.serial_number + "_read_start": t_start,
                self.serial_number + "_read_end": t_end,
                self.serial_number + "_frame_received": frames.get_timestamp()
            }
            print(f"timestamp_dict: {timestamp_dict}")

            data_dict = {"image": {self.serial_number: self._process_frame(color_frame)}}
            return data_dict, timestamp_dict

        except RuntimeError as e:
            print(f"Error reading frames from RealSense {self.serial_number}: {e}")
            return None

    def stop_recording(self):
        self.recording_active = False
        if not self.frame_buffer: return

        print(f"[{self.serial_number}] Saving {len(self.frame_buffer)} frames...")
        os.makedirs(self.recording_dir, exist_ok=True)

        timestamps = [f[0] for f in self.frame_buffer]
        frames = [f[1] for f in self.frame_buffer]

        if frames:
            # FIX 3: Calculate Actual FPS based on recorded duration
            duration = timestamps[-1] - timestamps[0]
            actual_fps = len(frames) / duration if duration > 0 else self.fps
            print(f"Detected FPS: {actual_fps:.2f} (Target was {self.fps})")

            ts_path = os.path.join(self.recording_dir, f"{self.serial_number}_timestamps.npy")
            np.save(ts_path, np.array(timestamps, dtype=np.float32))

            h, w = frames[0].shape[:2]
            vid_path = os.path.join(self.recording_dir, f"{self.serial_number}_color.mp4")
            
            # Use actual_fps to ensure playback speed matches reality
            vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), actual_fps, (w, h))
            for f in frames:
                vw.write(f)
            vw.release()
            print(f"[{self.serial_number}] Saved successfully.")

        self.frame_buffer = []


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