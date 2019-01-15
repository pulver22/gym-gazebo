import rospy
import numpy as np, quaternion
import math

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState


class NavigationUtilities():
    def __init__(self, min_x, max_x, min_y, max_y, reference_frame, model_name,
               distance, acceptance_distance, offset, positive_reward):
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.max_x = max_x
        self.reference_frame = reference_frame
        self.model_name = model_name
        self.acceptance_distance = acceptance_distance
        self.offset = offset
        self.positive_reward = positive_reward



    def getRandomPosition(self):
        random_pose = ModelState()

        tmp_x = np.random.uniform(low=self.min_x, high=self.max_x)
        tmp_y = np.random.uniform(low=self.min_y, high=self.max_y)
        random_pose.pose.position.x = tmp_x
        random_pose.pose.position.y = tmp_y
        random_pose.pose.position.z = 0.0
        # print("Random position (X,Y)= (" + str(tmp_x) + "," + str(tmp_y)+ ")")
        yaw = np.random.uniform(low=0, high=360)
        # quaternion = quaternion_from_euler(roll=0, pitch=0, yaw=yaw)  # NOTE: Py3 does not support tf
        orientation = quaternion.from_euler_angles(0.0, 0.0, yaw)  # roll, pitch, yaw
        random_pose.pose.orientation.x = orientation.components[1]
        random_pose.pose.orientation.y = orientation.components[2]
        random_pose.pose.orientation.z = orientation.components[3]
        random_pose.pose.orientation.w = orientation.components[0]

        random_pose.reference_frame = self.reference_frame
        random_pose.model_name = self.model_name

        return random_pose



    def getRandomTargetPosition(self, initial_pose):
        """
        Generate a random target within the arena
        :return:
        """
        low = initial_pose.pose.position.x - self.offset
        high = initial_pose.pose.position.x + self.offset
        target_x = np.random.uniform(low=low, high=high)
        low = initial_pose.pose.position.y - self.offset
        high = initial_pose.pose.position.y + self.offset
        target_y = np.random.uniform(low=low, high=high)
        # target_x = np.random.uniform(low=self.min_x, high=self.max_x)
        # target_y = np.random.uniform(low=self.min_y, high=self.max_y)
        target_x = 0.0
        target_y = 0.0
        target_z = 0.0
        return  np.array((target_x, target_y, target_z))

    def getRobotAbsPose(self):
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        robot_abs_pose = model_coordinates(self.model_name, self.reference_frame)

    def getGoalDistance(self, robot_abs_pose, target_position):
        """
        Calculate the distance between two points in space
        :param goal_a:
        :param goal_b:
        :return:
        """
        position = np.array(
            (robot_abs_pose.pose.position.x, robot_abs_pose.pose.position.y,
             robot_abs_pose.pose.position.z))

        assert position.shape == target_position.shape
        # print("goal_distance", np.linalg.norm(goal_a - goal_b, axis=-1))
        return np.linalg.norm(position - target_position, axis=-1)

    def getReward(self, distance):
        """
        Calculate the reward as euclidean distance from robot to target
        :return:
        """

        # if self.distance < self.proximity_distance:
        if distance < self.acceptance_distance:
            return self.positive_reward
        #   else:
        #      return self.positive_reward * 0.01 - self.distance
        else:
            return - distance.astype(np.float32) * 0.1

    def getBearingEuler(self, robot_abs_pose):
        """
        Get the robot absolute bearing to the goalself.
        """
        x = robot_abs_pose.pose.orientation.x
        y = robot_abs_pose.pose.orientation.y
        z = robot_abs_pose.pose.orientation.z
        w = robot_abs_pose.pose.orientation.w

        q = np.quaternion(x, y, z, w)# arg is expressed in quaternion
        euler_bearing = quaternion.as_euler_angles(q)
        # FIX: self.euler_beraing[1] is in [0,pi] while self.euler_bearing[0] is in [-pi, pi] and gives the sign
        if z*w < 0:
            euler_bearing[1] = 2 * math.pi - self.euler_bearing[1]  # cast to [0, 2pi]
        # print("Quaternion: ", str(self.robot_abs_pose.pose.orientation))
        # print(" -> Euler: ", str(self.euler_bearing[1]))
        return euler_bearing

    def getRobotTargetAbsAngle(self, robot_abs_pose, target_position):
        """
        Calculate the angle (in degreed) between the X-axis and the vector connecting the robot to its target
        :return:
        """
        delta_y = robot_abs_pose.pose.position.y - target_position[1]
        delta_x = robot_abs_pose.pose.position.x - target_position[0]
        # cos_B = delta_x / (math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2)))
        # sin_B = delta_y / (math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2)))
        robot_target_abs_angle = math.atan2(delta_y, delta_x)
        # print("Delta_x: ", str(delta_x), " delta_y: ", str(delta_y))
        # print("[1]", str(self.robot_target_abs_angle))
        if delta_y <= 0:
            robot_target_abs_angle = robot_target_abs_angle + 2 * math.pi
        # print("[2]", str(self.robot_target_abs_angle))
        return robot_target_abs_angle * 180 / math.pi
        # print("[3]", str(self.robot_target_abs_angle))

    def getRobotRelOrientation(self, robot_target_abs_angle, robot_abs_bearing):
        """
        Get the relative angle between the robot orientation and the vector connecting the robot to its target
        :return:
        """
        sign = -1

        robot_rel_orientation = abs(robot_target_abs_angle - (robot_abs_bearing * 180 / 3.14))
        # if self.robot_target_abs_angle >= 180:
        # Make the If statement more readable

        alpha = robot_abs_bearing * 180 / 3.14
        beta = robot_target_abs_angle
        # print("Beta: ", str(beta))
        # print("Alpha: ", str(alpha))
        # If the following condition applies, apply a positive sign to the angle in order to distinguish rotation clockwise(-) from anticlockwise (+)

        # if (0.0 <= beta <= 90.0) and (beta <= alpha <= (180.0 + beta)) :
        #     sign = 1
        #     rospy.logerr("[1]Signed changed!")
        # elif (90.0 <= beta <= 180.0) and (beta <= alpha <= 360.0 - (180.0 - beta)):
        #     sign = 1
        #     rospy.logerr("[2]Signed changed!")
        # elif (180.0 <= beta <= 270.0) and (beta <= alpha <= 360.0) or (0.0 <= alpha <= (beta - 180.0)):
        #     sign = 1
        #     rospy.logerr("[3]Signed changed!")
        # elif (270.0 <= beta <=  360.0) and (-(360.0 - beta) <= alpha <= 90.0 + (360.0 - beta)):
        #     sign = 1
        #     rospy.logerr("[4]Signed changed!")
        # else:
        if (0.0 <= beta <= 90.0) and (beta <= alpha <= (180.0 + beta)) or \
                ((90.0 <= beta <= 180.0) and (beta <= alpha <= 360.0 - (180.0 - beta))) or \
                ((180.0 <= beta <= 270.0) and (beta <= alpha <= 360.0) or (0.0 <= alpha <= (beta - 180.0))) or \
                ((270.0 <= beta <= 360.0) and (-(360.0 - beta) <= alpha <= 90.0 + (360.0 - beta))):
            sign = 1
        #     rospy.logerr("Signed changed!")
        # print("No sign changed")
        return sign * abs(180 - robot_rel_orientation)
        # if self.robot_target_abs_angle >= 180:
        #     self.robot_rel_orientation = abs(180 - self.robot_rel_orientation)

        # print("Robot relative orientation: ", str(self.robot_rel_orientation))