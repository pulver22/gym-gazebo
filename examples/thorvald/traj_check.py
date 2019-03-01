import rospy
import time
import os

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_srvs.srv import Empty


vel_pub = rospy.Publisher('nav_vel', Twist, queue_size=10)
vel_msg = Twist()
vel_msg.linear.x = 1.0
vel_msg.angular.z = 0.0

model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

rospy.init_node('traj_checker')



rospy.wait_for_service('/gazebo/unpause_physics')
try:
    unpause()
    # print("UnPausing")
except (rospy.ServiceException) as e:
    print("/gazebo/unpause_physics service call failed")


rospy.sleep(2.0)
# r = rospy.Rate(10)  # 10Hz


robot_abs_pose = model_coordinates('thorvald_ii', 'world')
init_pos = robot_abs_pose.pose.position.x

# while not rospy.is_shutdown():

s_start = rospy.get_rostime().secs
n_start = rospy.get_rostime().nsecs
r_start = time.time()
start = rospy.get_rostime()
# start = float(str(s_start) + "." + str(n_start))
# print("Start: ", start)
timer = rospy.get_rostime()
# print("START: ", str(start))
# rospy.sleep(rospy.Duration(0.5))
msg_counter = 0


###############################
# while (rospy.get_rostime().secs - s_start <= 5.0):
#     msg_counter += 1
vel_pub.publish(vel_msg)
rospy.sleep(rospy.Duration(0, 500000000))


#
# vel_msg.linear.x = 1.0
# vel_msg.angular.z = -0.8
# #
# vel_pub.publish(vel_msg)
# rospy.sleep(5.0)
#
stop = rospy.get_rostime()
r_stop = time.time()
#
vel_msg.linear.x = 0.0
vel_msg.angular.z = 0.0

vel_pub.publish(vel_msg)
rospy.sleep(2.0)


rospy.wait_for_service('/gazebo/pause_physics')
try:
    pause()
    # print("Pausing")
except (rospy.ServiceException) as e:
    print("/gazebo/pause_p hysics service call failed")


print("SRostime: ", abs(stop.secs - s_start))
print("NRostime: ", abs(stop.nsecs - n_start))
print("Wall time: ", r_stop - r_start)


# Get robot pose at the end of the velocity command
robot_abs_pose = model_coordinates('thorvald_ii', 'world')
final_pos = robot_abs_pose.pose.position.x
print("Travelled distance: ", final_pos - init_pos)
print("Init_pose_x: ", init_pos)
print("Final_pose_x: ", final_pos)
print("Message sent: ", msg_counter)