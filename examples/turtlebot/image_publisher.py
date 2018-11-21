import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats, HeaderString
from std_msgs.msg import UInt8MultiArray, Byte

import numpy as np

observation = None


def observation_callback( message):
    # print("Callback")
    global observation
    observation = message
    return

def rebuild_callback(message):
    print (np.fromstring(message.data, dtype=int))

def publish_grey_image():
    obs_message = None
    bridge = CvBridge()
    # print("Camera Empty: ", obs_message)
    while obs_message is None:
        try:
            obs_message = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            # obs_message = observation
        except:
            print("Error")
    # Convert from sensor_msgs::Image to cv::Mat
    cv_image = bridge.imgmsg_to_cv2(obs_message, desired_encoding="passthrough")
    # Access global variable and store image as numpy.array
    obs_message = np.asarray(cv_image)
    # print("Type:", type(obs_message))
    # print("Type data:", type(obs_message.data))
    # print("Image: ", obs_message)
    cv_image = cv2.cvtColor(obs_message, cv2.COLOR_BGR2GRAY)
    cv_image = cv2.resize(cv_image, (84, 84))
    obs_message = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
    print("Type:", obs_message)
    return obs_message


def main():
    rospy.init_node('DRL')
    r = rospy.Rate(10)
    rospy.Subscriber('/camera/rgb/image_raw', Image, observation_callback)
    # rospy.Subscriber('/drl/camera', HeaderString, rebuild_callback)
    pub = rospy.Publisher('/drl/camera', Byte, queue_size=5)

    # publish_grey_image()
    # msg = HeaderString()
    msg = Byte

    while not rospy.is_shutdown():
        msg.data = publish_grey_image().tostring()
        pub.publish(msg)
        r.sleep()


if __name__ == "__main__":
    main()
