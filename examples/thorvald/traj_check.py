import rospy
import os

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_srvs.srv import Empty


vel_pub = rospy.Publisher('nav_vel', Twist, queue_size=10)

rospy.init_node('traj_checker')
rospy.sleep(2)
r = rospy.Rate(10)  # 10Hz

vel_msg = Twist()
vel_msg.linear.x = 0.5
vel_msg.angular.z = 0.0

print("Vel_msg: ", str(vel_msg))
model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)



# while not rospy.is_shutdown():
start = rospy.get_rostime().nsecs
print("Start: ", start)
timer = rospy.get_rostime()
# print("START: ", str(start))
# rospy.sleep(rospy.Duration(0.5))
msg_counter = 0
# while(not rospy.is_shutdown()):
# while (abs(rospy.get_rostime().nsecs - start) <= 500000000 ):
#     # print("ct: ", rospy.get_rostime().nsecs)
#     # if(abs(rospy.get_rostime().nsecs - start) <= 5e9):
#         # print(" -> ", str(rospy.get_rostime().secs))
#     vel_pub.publish(vel_msg)
#     msg_counter = msg_counter + 1
#         # print(msg_counter)
#     # else:
#     #     break
#     # r.sleep()
##############################
# rospy.wait_for_service('/gazebo/unpause_physics')
# try:
#     unpause()
#     # print("UnPausing")
# except (rospy.ServiceException) as e:
#     print("/gazebo/unpause_physics service call failed")
###############################
while (msg_counter <= 50000 ):
    vel_pub.publish(vel_msg)
    msg_counter +=1
print("Rostime: ", abs(rospy.get_rostime().nsecs - timer.nsecs)*1e-9)
rospy.sleep(1.0)
###############################
# rospy.wait_for_service('/gazebo/pause_physics')
# try:
#     pause()
#     # print("Pausing")
# except (rospy.ServiceException) as e:
#     print("/gazebo/pause_p hysics service call failed")
###############################

    # rospy.sleep(rospy.Duration(0.5))
# os.system("rostopic pub -1 /nav_vel geometry_msgs/Twist  '{linear:  {x: 10.0, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'")
# rospy.sleep(rospy.Duration(5.0))
    # r.sleep()
# print("STOP")
# vel_msg.linear.x = 1.0
# vel_msg.angular.z = -0.8
# start = rospy.get_rostime().secs
# while(rospy.get_rostime().secs - start < 5):
#     # print(" -> ", str(rospy.get_rostime().secs))
#     vel_pub.publish(vel_msg)

# Get robot pose at the end of the velocity command
robot_abs_pose = model_coordinates('thorvald_ii', 'world')
print("Final position: ", str(robot_abs_pose))
print("Message sent: ", msg_counter)