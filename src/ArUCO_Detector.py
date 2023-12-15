#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

MARKER_SIZE = 20.0
LEFT_CAM_HD = {
    'fx': 533.895,
    'fy': 534.42,
    'cx': 642.65,
    'cy': 349.4705,
    'k1': -0.0557809,
    'k2': 0.0279374,
    'p1': 0.000647675,
    'p2': -0.000394777,
    'k3': -0.0106177
}
CAM_MAT = np.array([[LEFT_CAM_HD['fx'], 0, LEFT_CAM_HD['cx']],
                    [0, LEFT_CAM_HD['fy'], LEFT_CAM_HD['cy']],
                               [0, 0, 1]])
DIST_COEF = np.array([-0.0557809, 0.0279374, 0.000647675, -0.000394777, -0.0106177])
MARKER_DICT = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
PARAM_MARKERS = cv.aruco.DetectorParameters_create()
ARUCO_LIST = []

def detect_markers(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = cv.aruco.detectMarkers(gray, MARKER_DICT, parameters=PARAM_MARKERS)
    return marker_corners, marker_IDs

def estimate_pose(marker_corners):
    rVec, tVec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, CAM_MAT, DIST_COEF)
    return tVec

def ArucoFunc(frame):
    marker_corners, marker_IDs = detect_markers(frame)
    aruco_list = []

    if marker_corners:
        tVec = estimate_pose(marker_corners)

        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            distance = np.sqrt(tVec[i][0][2] * 2 + tVec[i][0][0] * 2 + tVec[i][0][1] ** 2)

            # Add marker information to the list, including Z-coordinate
            aruco_list.append({
                "id": ids[0],
                "x-coordinate": tVec[i][0][0],
                "y-coordinate": tVec[i][0][1],
                "z-coordinate": tVec[i][0][2],
                "distance": distance,
                "visited": True
            })
    return aruco_list
    
def callback(msg):
    try:
        # Convert ROS Image message to OpenCV image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ARUCO_LIST.extend(ArucoFunc(cv_image))
    except Exception as e:
        rospy.logerr(f"Error processing image: {repr(e)}")

def listener():
    rospy.init_node("ArUCO Detector",anonymous=True)
    rospy.Subscriber("/zed/image_left",Image, callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        listener()
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {repr(e)}")
        rospy.signal_shutdown("Unexpected error occurred")