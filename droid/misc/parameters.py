import os
from cv2 import aruco

# Robot Params #
nuc_ip = "10.0.0.3"
robot_ip = "10.0.0.2"
laptop_ip = "10.0.0.1"
sudo_password = "kovan123"
robot_type = "panda"  # 'panda' or 'fr3'
robot_serial_number = "295341-1324423"

# Camera ID's #
hand_camera_id = ""
varied_camera_1_id = ""
varied_camera_2_id = ""

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = "C14KZhAZx5JVyqCXnu6D7Ahy5cCGRh"

# Code Version [DONT CHANGE] #
droid_version = "1.3"

