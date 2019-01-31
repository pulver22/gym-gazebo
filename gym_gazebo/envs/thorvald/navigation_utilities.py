import rospy
import numpy as np, quaternion
import math

from gazebo_msgs.msg import ModelState,  ModelStates
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Twist


class NavigationUtilities():
    def __init__(self, min_x, max_x, min_y, max_y, reference_frame, model_name,
               proximity_distance, acceptance_distance, offset, positive_reward):
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.max_x = max_x
        self.reference_frame = reference_frame
        self.model_name = model_name
        self.acceptance_distance = acceptance_distance
        self.proximity_distance = proximity_distance
        self.offset = offset
        self.positive_reward = positive_reward



    def getRandomRobotPosition(self):
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

    def get_velocity_message(self, action):
        """
        Helper function.
        Wraps an action vector into a Twist message.
        """
        # Set up a Twist message to publish.
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        return action_msg

    def get_velocity_message_discrete(self, action):
        """
        Helper function.
        Wraps an action vector into a Twist message.
        """
        # Set up a Twist message to publish.
        action_msg = Twist()
        if action == 0:
            action_msg.linear.x = 0.2
        elif action == 1:
            action_msg.angular.z = -0.2
        elif action == 2:
            action_msg.angular.z = 0.2
        return action_msg

    def calculate_collision_from_lidar(self,data):
        """
        Read the Lidar data and return done = True with a penalization is an obstacle is perceived within the safety distance
        :param data:
        :return:
        """
        done = False
        reward = 0.0
        for i, item in enumerate(data):

            # If the laser reading return an infinite distance, clip it to 100 meters
            if (data[i] == np.inf):
                data[i] = 100

            # If the laser reading returns not a number, clip it to 0 meters
            if np.isnan(data[i]):
                data[i] == 0

            # If the obstacles is closer than the minimum safety distance, stop the episode
            if (self.min_range > data[i] > 0):
                rospy.logerr("Collision detected")
                return True, self.penalization
        return done, reward

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
        return model_coordinates(self.model_name, self.reference_frame)

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
            euler_bearing[1] = 2 * math.pi - euler_bearing[1]  # cast to [0, 2pi]
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
        print("-----")
        print("Beta: ", str(beta))
        print("Alpha: ", str(alpha))
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
                ((270.0 <= beta <= 360.0) and (-(360.0 - beta) <= alpha <= abs(180.0 - (360.0 - beta)))):
            sign = 1
            rospy.logerr("Signed changed!")
        # print("No sign changed")
        return sign * abs(180 - robot_rel_orientation)
        # if self.robot_target_abs_angle >= 180:
        #     self.robot_rel_orientation = abs(180 - self.robot_rel_orientation)

        # print("Robot relative orientation: ", str(self.robot_rel_orientation))

    def normalise(self, value, min, max):
        """
        Return a normalised value
        :param value:
        :param min:
        :param max:
        :return:
        """
        return (value - min)/(max - min)

    def getCollisionPenalty(self, last_collision):
        """
        Check for collision and penalise them
        :return:
        """
        penalty = 0.0
        if last_collision != None:
            for wrench in last_collision.wrenches:
                penalty += min(0.01 * math.sqrt(pow(wrench.force.x,2) + pow(wrench.force.y,2) + pow(wrench.force.z,2)), 10)

        return -penalty

    def checkRobotPose(self, robot_pose):
        """
        Check if the robot is within an object
        :return:
        """
        objects_msg = None
        while objects_msg == None:
            objects_msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)


        for i in range(0, len(objects_msg.name)):
            # print("Checking pose ", (i+1), "of", len(objects_msg.name))
            if (objects_msg.pose[i].position.x - self.proximity_distance < robot_pose.pose.position.x < objects_msg.pose[i].position.x + self.proximity_distance):
                if (objects_msg.pose[i].position.y - self.proximity_distance < robot_pose.pose.position.y < objects_msg.pose[i].position.y + self.proximity_distance):
                    return False

        return True

    def controller(self,goal_info):
        """
        Generate an action given the observation
        :param goal_info: an ndarray containing distance from goal and relative angle
        :return:
        """
        action = Twist()
        # if the sin is positive, the robot must rotate towards its left
        action.linear.x = 0.2
        if goal_info[2] > 0.5:
            action.angular.z = 0.2
            action_str = "Left"
        else:
            action.angular.z = -0.2
            action_str = "Right"
        print("[C] Action: ", action_str)
        return action
