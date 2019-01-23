import gym
import rospy
import roslib
import roslaunch
import time
import numpy as np, quaternion
import cv2
import math

from gym import utils, spaces
from gym.utils import seeding

from gym_gazebo.envs import gazebo_env

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ContactState
from rosgraph_msgs.msg import Clock
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats, HeaderString

from cv_bridge import CvBridge, CvBridgeError

from gym_gazebo.envs.thorvald.navigation_utilities import NavigationUtilities




class GazeboThorvaldCameraCnnPPOEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboThorvald.launch")
        self.vel_pub = rospy.Publisher('nav_vel', Twist, queue_size=5)
        # self._drl_sub = rospy.Subscriber('/drl/camera', numpy_msg(HeaderString), self.observation_callback)
        self.camera_sub = rospy.Subscriber('/thorvald_ii/kinect2/hd/image_color_rect', Image, self.observation_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clock_callback)
        self.collision_sub = rospy.Subscriber('/fag', ContactState, self.contact_callback)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_position_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self._observation_msg = None
        self._lidar_msg = None
        self._last_obs_header = None
        self._last_lidar_header = None
        self.obs = None
        self.reward = 0
        self.done = False
        self.max_episode_steps = 500  # limit the max episode step
        self.iterator = 0  # class variable that iterates to accounts for number of steps per episode
        self.reset_position = True
        self.use_cosine_sine = True

        # Action space
        self.velocity_low = np.array([0.0, -0.2], dtype=np.float32)
        self.velocity_high = np.array([0.3, 0.2], dtype=np.float32)
        self.action_space = spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32)

        # Camera setting
        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1

        # Lidar setting
        self.min_range = 0.5

        # Observation space
        # self.observation_high = 255.0
        # self.observation_low = 0.0
        self.observation_high = 1.0  # DEBUG: mockup images
        self.observation_low = 0.0  # DEBUG: mockup images
        # self.observation_space = spaces.Box(self.observation_low, self.observation_high, shape=(self.img_rows, self.img_cols, self.img_channels), dtype=np.uint8)  # Without goal info
        self.goal_info = np.zeros(shape=(self.img_rows, 1, 1))  # Arrays need to have same dimesion in order to be concatened
        self.observation_space = spaces.Box(low=self.observation_low,
                                            high=self.observation_high,
                                            shape=(self.img_rows, self.img_cols + 1, self.img_channels),
                                            dtype=np.float16)  # With goal info


        # Environment hyperparameters
        self.initial_pose = None
        self.min_x = -4.0
        self.max_x = 4.0
        self.min_y = - 4.0
        self.max_y = 4.0
        self.offset = 3.0
        self.max_distance = 15.0
        self.target_position = [None, None]
        self.model_name = 'thorvald_ii'
        self.reference_frame = 'world'
        self.last_clock_msg = None
        self.last_step_ts = None
        self.skip_time = 500000000  # expressed in nseconds
        self.synch_time = True



        self.reward_range = (-1000.0, 1000)
        self.penalization = - 200
        self.positive_reward = 800
        self.acceptance_distance = 1.0
        self.proximity_distance = 0.5
        self.distance = None
        self.robot_abs_pose = None
        self.robot_rel_orientation = None
        self.robot_target_abs_angle = None
        self.euler_bearing = None
        self.last_collision = None

        self.nav_utils = NavigationUtilities(min_x=self.min_x, max_x=self.max_x, min_y=self.min_y, max_y=self.max_y,
                                             reference_frame=self.reference_frame, model_name=self.model_name,
                                             distance=self.distance, acceptance_distance=self.acceptance_distance,
                                             offset=self.offset, positive_reward=self.positive_reward)

        self.time_start = 0.0
        self.time_stop = 0.0
        self.rospy_time_start = 0.0
        self.rospy_time_stop = 0.0

        self.r = rospy.Rate(20)

        self._seed()

    def clock_callback(self, message):
        """
        Callback method for the subscriber of the clock topic
        :param message:
        :return:
        """
        # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
        self.last_clock_msg = int(message.clock.nsecs)
        # print(self.last_clock_msg)

    def observation_callback(self, message):
        """
        Callback method for the subscriber of the camera
        """
        if message.header.seq != self._last_obs_header:
            self._last_obs_header = message.header.seq
            self._observation_msg = message
        else:
            ROS_ERROR("Not receiving images")

    def lidar_callback(self, message):
        """
        Callback method for the subscriber of lidar
        """
        if message.header.seq != self._last_lidar_header:
            self._last_lidar_header = message.header.seq
            self._lidar_msg =  np.array(message.ranges)
        else:
            ROS_ERROR("Not receiving lidar readings")

    def contact_callback(self, message):
        """
        Parse ContactState messages for possible collision (filter those involving the ground)
        :param message:
        :return:
        """

        if "ground_plane" in message.collision1_name or "ground_plane" in message.collision2_name:
            pass
        elif "thorvald_ii" in message.collision1_name or "thorvald_ii" in message.collision2_name:
            self.last_collision = message


    def take_observation(self):
        """
        Take observation from the environment and return itself.
        """

        obs_message = None
        bridge = CvBridge()
        # print("Camera Empty: ", obs_message)
        while obs_message is None:
            try:
                obs_message = self._observation_msg
            except:# CvBridgeError as ex:
                rospy.logerr ("ERROR!!")#, ex)
        # Convert from sensor_msgs::Image to cv::Mat
        cv_image = bridge.imgmsg_to_cv2(obs_message, desired_encoding="bgr8")
        # Convert the image to grayscale
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Resize and reshape the image according to the network input size
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        obs_message = cv_image.reshape( cv_image.shape[0], cv_image.shape[1], 1)
        # print("  --> Observation acquired")
        return obs_message

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

    def calculate_collision(self,data):
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

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - reward
            - done (status)
            - action
            - observation
            - dictionary (#TODO clarify)
        """
        self.iterator += 1
        # print("[B]Time difference between step: ", (float(time.time()) - self.time_stop), " sec")
        # print("[B]ROSPY Time difference between step: ", abs(rospy.get_rostime().nsecs - self.rospy_time_stop)*1e-9, " sec")
        # print("[B]ROSPY Time ", rospy.get_time())

        self.time_stop = float(time.time())
        self.rospy_time_stop = float(rospy.get_rostime().nsecs)


        #########################
        ##       ACTION        ##
        #########################
        if self.synch_mode == True:
            # Unpause simulation
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
                # print("UnPausing")
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

        if self.synch_mode == True:
            # Pause simulation
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                self.pause()
                # print("Pausing")
            except (rospy.ServiceException) as e:
                print("/gazebo/pause_p hysics service call failed")

        #########################
        ##         STATE       ##
        #########################
        self.ob = None
        while (self.ob is None):
            try:
                # print("  --> Acquiring observation")
                # self.ob = self.take_observation()
                self.ob = np.ones(shape=(84, 84, 1))  # DEBUG
            except:
                rospy.logerr("Problems acquiring the observation")

        # Calculate actual distance from robot
        self.robot_abs_pose = self.nav_utils.getRobotAbsPose()
        self.distance = self.nav_utils.getGoalDistance(self.robot_abs_pose, self.target_position)
        self.goal_info[0] = self.distance
        self.euler_bearing = self.nav_utils.getBearingEuler(self.robot_abs_pose)
        # self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        self.robot_target_abs_angle = self.nav_utils.getRobotTargetAbsAngle(self.robot_abs_pose, self.target_position)
        self.robot_rel_orientation = self.nav_utils.getRobotRelOrientation(self.robot_target_abs_angle,
                                                                           self.euler_bearing[1])
        self.goal_info[1] = self.robot_rel_orientation
        if self.use_cosine_sine == True:
            self.goal_info[1] = math.cos(
                self.robot_rel_orientation * 3.14 / 180.0)  # angles must be expressed in radiants
            self.goal_info[2] = math.sin(self.robot_rel_orientation * 3.14 / 180.0)
            # Normalise the sine and cosine
            self.goal_info[1] = self.nav_utils.normalise(value=self.goal_info[1], min=-1.0, max=1.0)
            self.goal_info[2] = self.nav_utils.normalise(value=self.goal_info[2], min=-1.0, max=1.0)
        # Include also contact information
        self.goal_info[3] = self.last_collision.wrenches[0].force.x
        self.goal_info[4] = self.last_collision.wrenches[0].force.y
        self.goal_info[5] = self.last_collision.wrenches[0].force.z

        # Append the goal information (distance and bearing) to the observation space
        self.ob = np.append(self.ob, self.goal_info, axis=1)

        #########################
        ##        REWARD       ##
        #########################
        self.reward = self.nav_utils.getReward(self.distance)

        #########################
        ##         DONE        ##
        #########################
        # If the reward is greater than zero, we are in proximity of the goal. So Done needs to be set to true
        if self.reward >= 0 or self.iterator == (self.max_episode_steps - 1):
            self.done = True
        # Printing info
        if self.done == True:
            print("Final distance to goal: ", self.distance)
            if self.reward >= 0:
                rospy.logwarn("The robot reached its target")
            else:
                pass

        return self.ob, self.reward, self.done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        print("New episode")

        # Reset the step iterator
        self.iterator = 0

        f
        self.synch_mode == True:
        # Unpause simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
            # print("UnPausing")
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")


        # Reset robot position
        if self.reset_position is True:
            # rospy.wait_for_service('/gazebo/reset_simulation')
            rospy.wait_for_service('/gazebo/set_model_state')
            self.initial_pose = self.getRandomPosition()
            #print("New initial position: ", new_initial_pose)
            try:
                self.set_position_proxy(self.initial_pose)
                # self.reset_proxy()
            except (rospy.ServiceException) as e:
                rospy.logerr ("/gazebo/reset_simulation service call failed")
                rospy.logerr("/gazebo/set_model_state service call failed")

        # Take an observation
        self.ob = None
        while (self.ob is None):
            try:
                # print("  --> Acquiring observation")
                self.ob = self.take_observation()
            except:
                rospy.logerr("Problems acquiring the observation")

        # Reset the target position and calculate distance robot-target
        self.target_position = self.nav_utils.getRandomTargetPosition(self.initial_pose)
        self.robot_abs_pose = self.nav_utils.getRobotAbsPose()
        self.distance = self.nav_utils.getGoalDistance()
        self.goal_info[0] = self.distance

        self.euler_bearing = self.nav_utils.getBearingEuler(self.robot_abs_pose)
        # self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        self.robot_target_abs_angle = self.nav_utils.getRobotTargetAbsAngle(self.robot_abs_pose, self.target_position)
        self.robot_rel_orientation = self.nav_utils.getRobotRelOrientation(self.robot_target_abs_angle,
                                                                           self.euler_bearing[1])
        self.goal_info[1] = self.robot_rel_orientation
        if self.use_cosine_sine == True:
            self.goal_info[1] = math.cos(
                self.robot_rel_orientation * 3.14 / 180.0)  # angles must be expressed in radiants
            self.goal_info[2] = math.sin(self.robot_rel_orientation * 3.14 / 180.0)
            # Normalise the sine and cosine
            self.goal_info[1] = self.normalise(value=self.goal_info[1], min=-1.0, max=1.0)
            self.goal_info[2] = self.normalise(value=self.goal_info[2], min=-1.0, max=1.0)

        # Append the goal information (distance and bearing) to the observation space
        self.ob = np.append(self.ob, self.goal_info, axis=1)

        if self.synch_mode == True:
            # Pause the simulation
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                self.pause()
                # print("Pausing")
            except (rospy.ServiceException) as e:
                print("/gazebo/pause_physics service call failed")


        # Printing info
        print("New starting position: [", self.initial_pose.pose.position.x, ",", self.initial_pose.pose.position.y, ", ",
              self.initial_pose.pose.position.z, "]")
        print("New target: ", self.target_position)
        print("Distance to goal: ", self.distance)

        self.last_step_ts = self.last_clock_msg
        self.rospy_time_stop = float(rospy.Time.now().nsecs)
        self.done = False

        return self.ob