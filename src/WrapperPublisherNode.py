#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import pyzed.sl as sl
from cv_bridge import CvBridge
import cv2 as cv
def init_zed():
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        rospy.logerr("Failed to open ZED camera: %s", repr(status))
        rospy.signal_shutdown("Failed to open ZED camera")
        exit(-1)
    rospy.loginfo("ZED camera started successfully!")
    return zed

def init():
    pub = rospy.Publisher("/zed/image_left",Image,queue_size=10)
    rospy.init_node("ZED-ROS Wrapper",anonymous=True)
    rate = rospy.Rate(5)

    rospy.loginfo("Publisher Zed_Wrapper_Publisher has started")

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    zed = init_zed()

    bridge = CvBridge()

    while not rospy.is_shutdown():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image,sl.VIEW.LEFT)
            image_cv = image.get_data()
            image_cv = cv.cvtColor(image_cv, cv.COLOR_RGBA2BGR)
            image_ros = bridge.cv2_to_imgmsg(image_cv, encoding="bgr8")
            pub.publish(image_ros)
        else:
            print("Zed couldn't grab the image")
        rate.sleep()

if __name__ == "__main__":
    try:
        init()
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {repr(e)}")
        # You can add additional cleanup actions or specific behavior here.
        rospy.signal_shutdown("Unexpected error occurred")

