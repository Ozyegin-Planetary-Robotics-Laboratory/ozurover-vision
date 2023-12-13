#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv2 import aruco
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

MARKER_SIZE = 17.3
left_cam_hd = {
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
cam_mat = np.array([[left_cam_hd['fx'], 0, left_cam_hd['cx']],
                               [0, left_cam_hd['fy'], left_cam_hd['cy']],
                               [0, 0, 1]])

dist_coef = np.array([-0.0557809, 0.0279374, 0.000647675, -0.000394777, -0.0106177])
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()
ArucoList = []

def ArucoFunc(frame,marker_dict,param_markers,cam_mat,dist_coef,list):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = cv.aruco.detectMarkers(gray, marker_dict,parameters=param_markers)
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            tag = {"id":ids,"x-coordinate":tVec[i][0][0],"y-coordiante":tVec[i][0][1],"visited":True}
            id_exists = any(tag["id"] == ids for tag in list)
            if not id_exists:
                # Append information to ArUcoList
                tag = {"id": ids[0], "x-coordinate": tVec[i][0][0], "y-coordinate": tVec[i][0][1], "visited": True}
                list.append(tag)
            distance = np.sqrt( tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2)
            
            # Draw the pose of the marker
            cv.putText(
            frame,
            f"id: {ids[0]} Dist: {distance}",
            top_right,
            cv.FONT_HERSHEY_PLAIN,
            1.3,
            (0, 0, 255),
            2,
            cv.LINE_AA)

            cv.putText(
            frame,
            f"x:{tVec[i][0][0]} y: {tVec[i][0][1]} ",
            bottom_right,
            cv.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 0, 255),
            2,
            cv.LINE_AA)
        
    #cv.imshow("anan", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        exit(-1)
    
def callback(data):
    try:
        # Convert ROS Image message to OpenCV image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        ArucoFunc(cv_image,marker_dict,param_markers,cam_mat,dist_coef,ArucoList)
        

    except Exception as e:
        rospy.logerr(f"Error processing image: {repr(e)}")


def listener():
    
    rospy.init_node("ArUCO Detector",anonymous=True)
    rospy.Subscriber("/zed/image_left",Image, callback)
    rospy.spin()
    for tag in ArucoList:
        print(tag)

if __name__ == "__main__":
    try:
        listener()
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {repr(e)}")
        # You can add additional cleanup actions or specific behavior here.
        rospy.signal_shutdown("Unexpected error occurred")