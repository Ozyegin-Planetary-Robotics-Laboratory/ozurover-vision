#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from ozurover_messages.msg import Marker

class Node:
    def __init__(self):
        print("slm")
        rospy.init_node("ares_aruco_detecter")
        self.pub = rospy.Publisher("ares/goal/marker",Marker,queue_size = 1)
        self.sub = rospy.Subscriber("/zed2i/zed_node/left/image_rect_color",Image,self.callback)
        self.MARKER_SIZE = 20.0
        self.CAM_MAT = np.array([[263.95489501953125, 0, 320],
                         [0, 263.95489501953125, 180],
                         [0, 0, 1]])    
    
        self.DIST_COEF = np.array([-0.784998, 2.44096, 0.000561938, -7.78445e-05, 0.113489, -0.680278, 2.29567, 0.281928])
        self.MARKER_DICT = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        self.PARAM_MARKERS = cv.aruco.DetectorParameters_create()
    

    def detect_markers(self,frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.marker_corners, self.marker_IDs, reject = cv.aruco.detectMarkers(gray, self.MARKER_DICT, parameters=self.PARAM_MARKERS)
        return self.marker_corners, self.marker_IDs
    

    def estimate_pose(self,marker_corners):
        self.rVec, self.tVec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, self.MARKER_SIZE, self.CAM_MAT, self.DIST_COEF)
        return self.tVec


    def aruco_func(self,frame):
        marker_corners, marker_IDs = self.detect_markers(frame)

        if marker_corners:


            
            for (markerCorner, markerID) in zip(marker_corners, marker_IDs):
                # extract the marker corners (which are always returned
                # in top-left, top-right, bottom-right, and bottom-left
                # order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # Convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # Draw the bounding box of the ArUco detection
                cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                # Compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the frame
                cv.putText(frame, str(markerID),
                            (topLeft[0], topLeft[1] - 15),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                

            tVec = self.estimate_pose(marker_corners)

            total_markers = range(0, marker_IDs.size)
            for ids, i in zip(marker_IDs, total_markers):
                aruco_tag = Marker()
                aruco_tag.pose.pose.position.x = tVec[i][0][0]
                aruco_tag.pose.pose.position.y = tVec[i][0][1]
                aruco_tag.pose.pose.position.z = tVec[i][0][2]
                aruco_tag.pose.header.frame_id = "zed2i_left_camera_frame"
                aruco_tag.type = ids[0]
                self.pub.publish(aruco_tag)

        frame_resized = cv.resize(frame,(1280,720),  interpolation=cv.INTER_LINEAR)
        cv.imshow("Detected Markers", frame_resized)
        cv.waitKey(1)   

    def callback(self, image_data):
        try:
            print("slm")
            # Convert ROS Image message to OpenCV image
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
            self.aruco_func(cv_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {repr(e)}")



if __name__ == "__main__":
    node = Node()
    rospy.spin()