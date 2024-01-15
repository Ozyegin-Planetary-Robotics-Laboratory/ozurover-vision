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
    def __init__(self):
        rospy.init_node("ares_aruco_detecter")
        """left_cam_hd = {'fx': 266.947509765625,
                       'fy': 267.2099914550781,
                       'cx': 321.32501220703125,
                       'cy': 174.73524475097656,
                       'k1': -0.7849979996681213,
                       'k2': 2.440959930419922,
                       'k3': 0.11348900198936462,
                       'p1': 0.0005619379808194935,
                       'p2': -7.784450281178579e-05}"""
        
        # ZED 2 Camera Information
        # Distortion Parameters
        #  D = [k1 k2 t1 t2 k3]
        # Camera Matrix
        #  K = [fx  0  cx]
        #      [0  fy  cy]
        #      [0   0   1]
        # Rectification Matrix
        #  R = 3x3 Identity Matrix
        # Projection Matrix
        #  P = [fx' 0 cx' Tx]
        #      [0 fy' cy' Ty]
        #      [0  0   1   0]
        
        self.pub = rospy.Publisher("ares/goal/marker", Marker, queue_size=10)
        self.sub = rospy.Subscriber("/zed2/left/image_rect_color", Image, self.callback)
        
        self._dist_coeff = np.array([-0.7849979996681213, 2.440959930419922, 0.0005619379808194935, -7.784450281178579e-05,0.11348900198936462])
        self._cam_matrix = np.array([[527.2972398956961, 0.0, 658.8206787109375],
                                     [0.0, 527.2972398956961, 372.25787353515625],
                                     [0.0, 0.0, 1.0]])
        
        self._marker_size = 20.0
        self._marker_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_1000)
        self._marker_params = cv.aruco.DetectorParameters_create()
    

    def detectMarkers(self,frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, self._marker_dict, parameters=self._marker_params)
        return corners, ids
    

    def estimatePose(self,marker_corners):
        _1, tVec, _2 = cv.aruco.estimatePoseSingleMarkers(marker_corners, self._marker_size, self._cam_matrix, self._dist_coeff)
        return tVec
    
    def callback(self, image_data):
        try:
            print("geldim")
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
            print("Image shape:", frame.shape)

            corners, ids = self.detectMarkers(frame)
            print(f"ids {ids}")
            print(f"corner {corners}")
            tVec = self.estimatePose(corners)
            print(f"tvec {tVec}")
            for ids, i in zip(ids, range(0, len(corners))):
                marker = Marker()
                marker.type = ids[0]
                marker.pose.pose.position.x = tVec[i][0][0]
                marker.pose.pose.position.y = tVec[i][0][1]
                marker.pose.pose.position.z = tVec[i][0][2]
                marker.pose.header.frame_id = "zed2i_left_camera_frame"
                self.pub.publish(marker)
                print(math.sqrt(tVec[i][0][0]**2+tVec[i][0][1]**2+tVec[i][0][2]**2))
        except Exception as e:
            rospy.logerr(f"Error processing image: {repr(e)}")


if __name__ == "__main__":
    node = Node()
    rospy.spin()
