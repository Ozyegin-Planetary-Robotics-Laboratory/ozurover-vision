#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import rospy
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ozurover_messages.msg import Marker
from geometry_msgs.msg import PoseStamped

class Node:
    def __init__(self, aruco_dict_type):
        rospy.init_node("ares_aruco_detector")

        # Load calibration data from the .npz file
        calib_data_path = "/home/aseris/catkin_ws/src/ares_autonomy/ozurover_vision/calibration/MultiMatrix.npz"
        calib_data = np.load(calib_data_path)

        self._cam_matrix = calib_data["camMatrix"]
        self._dist_coeff = calib_data["distCoef"]
        self._marker_size = 28.0
        self._marker_dict_type = aruco_dict_type
        self._marker_dict = cv.aruco.Dictionary_get(self._marker_dict_type)
        self._marker_params = cv.aruco.DetectorParameters_create()

        self.pub = rospy.Publisher("ares/goal/marker", Marker, queue_size=10)
        self.sub = rospy.Subscriber("/zed2/left/image_rect_color", Image, self.callback)

    def detect(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, self._marker_dict, parameters=self._marker_params)
        return corners, ids

    def estimate_pose(self, marker_corners):
        _, tVec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, self._marker_size, self._cam_matrix, self._dist_coeff)
        return tVec

    def callback(self, image_data):
        try:
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
            corners, ids = self.detect(frame)

            if corners:
                tVec = self.estimate_pose(corners)

                for i in range(len(ids)):
                    marker = Marker()
                    marker.type = ids[i][0]
                    marker.pose.pose.position.x = tVec[i][0][0]
                    marker.pose.pose.position.y = tVec[i][0][1]
                    marker.pose.pose.position.z = tVec[i][0][2]
                    marker.pose.header.frame_id = "zed2i_left_camera_frame"
                    print(f"ArUco {ids[i][0]} - x: {tVec[i][0][0]}, y: {tVec[i][0][1]}, z: {tVec[i][0][2]}")
                    self.pub.publish(marker)

        except Exception as e:
            rospy.logerr(f"Error processing image: {repr(e)}")

if __name__ == "__main__":
    # ArUco ölçülerini bir döngüde denemek için
    for aruco_dict_type in [ cv.aruco.DICT_5X5_50, cv.aruco.DICT_6X6_50,
                            cv.aruco.DICT_7X7_50, cv.aruco.DICT_ARUCO_ORIGINAL, cv.aruco.DICT_APRILTAG_16h5,
                            cv.aruco.DICT_4X4_250, cv.aruco.DICT_5X5_100, cv.aruco.DICT_6X6_100,
                            cv.aruco.DICT_7X7_100, cv.aruco.DICT_4X4_1000]:
        print(f"Testing ArUco Dictionary Type: {aruco_dict_type}")
    node = Node(cv.aruco.DICT_4X4_1000)
    rospy.spin()
