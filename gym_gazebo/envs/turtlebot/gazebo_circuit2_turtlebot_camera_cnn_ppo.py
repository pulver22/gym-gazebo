import gym
import rospy
import roslib
import roslaunch
import time
import numpy as np, quaternion
import cv2

from gym import utils, spaces
from gym.utils import seeding

from gym_gazebo.envs import gazebo_env

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats, HeaderString

from cv_bridge import CvBridge, CvBridgeError
#from tf.transformations import quaternion_from_euler




class GazeboCircuit2TurtlebotCameraCnnPPOEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2cTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        # self._drl_sub = rospy.Subscriber('/drl/camera', numpy_msg(HeaderString), self.observation_callback)
        self.camera_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.observation_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_position_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self._observation_msg = None
        self._lidar_msg = None
        self._last_obs_header = None
        self._last_lidar_header = None
        self.obs = None
        self.reward = None
        self.done = None
        self.action_space = None
        self.max_episode_steps = 1000  # limit the max episode step
        self.iterator = 0  # class variable that iterates to accounts for number of steps per episode
        self.reset_position = True

        # Action space
        self.velocity_low = np.array([0.0, -0.1], dtype=np.float32)
        self.velocity_high = np.array([0.2, 0.1], dtype=np.float32)
        self.action_space = spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32)
        self.last50actions = [0] * 50

        # Camera setting
        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1

        # Lidar setting
        self.min_range = 0.21

        # Observation space
        self.observation_high = 255
        self.observation_low = 0
        self.observation_space = spaces.Box(self.observation_low, self.observation_high, shape=(self.img_rows, self.img_cols, self.img_channels), dtype=np.uint8)

        # Environment hyperparameters
        self.min_x = -4.0
        self.max_x = 0
        self.min_y = - 7.5
        self.max_y = 0

        self.reward_range = (-np.inf, np.inf)


        self._seed()

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
                print ("ERROR!!")#, ex)
        # Convert from sensor_msgs::Image to cv::Mat
        cv_image = bridge.imgmsg_to_cv2(obs_message, desired_encoding="bgr8")
        # TODO: temporal fix, check image is not corrupted
        # if not (cv_image[self.img_rows // 2, self.img_cols // 2, 0] == 178 and cv_image[self.img_rows // 2, self.img_cols // 2, 1] == 178 and cv_image[
        #     self.img_rows // 2, self.img_cols // 2, 2] == 178):
        #     success = True
        # else:
        #     # pass
        #     print("/camera/rgb/image_raw ERROR, retrying")
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

    def calculate_done(self,data):
        done = False
        for i, item in enumerate(data):

            # If the laser reading return an infinite distance, clip it to 100 meters
            if (data[i] == np.inf):
                data[i] = 100

            # If the laser reading returns not a number, clip it to 0 meters
            if np.isnan(data[i]):
                data[i] == 0

            # If the obstacles is closer than the minimum safety distance, stop the episode
            if (self.min_range > data[i] > 0):
                done = True
        return done

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
        min_range = 0.2
        done = False
        self.iterator+=1
        # print("  --> Sending action")
        #TODO: Create an action message
        self.vel_pub.publish(self.get_velocity_message(action))
        # print("[", self.iterator, "] Action selected [lin, ang]: ", action)


        #########################
        ##         DONE        ##
        #########################
        laser_ranges = None
        while laser_ranges is None:
            try:
                # print("  --> Acquiring laser data")
                laser_ranges = self._lidar_msg
            except:
                pass

        self.done = self.calculate_done(laser_ranges)

        #########################
        ##         STATE       ##
        #########################
        self.ob = None
        while(self.ob is None):
            try:
                # print("  --> Acquiring observation")
                self.ob = self.take_observation()
            except:
                pass

        #########################
        ##        REWARD       ##
        #########################
        self.last50actions.pop(0) #remove oldest
        if action[0] > action[1]:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)
        action_sum = sum(self.last50actions)

        # Add center of the track reward
        laser_len = len(laser_ranges)
        left_sum = sum(laser_ranges[int(laser_len - (laser_len / 5)):int(laser_len - (laser_len / 10))])  # 80-90
        right_sum = sum(laser_ranges[int((laser_len / 10)):int((laser_len / 5))])  # 10-20
        center_detour = abs(right_sum - left_sum) / 5

        if not self.done:
            if action[0]:
                self.reward = 1 / float(center_detour+1)
            elif action_sum > 45: #L or R looping
                reward = -0.5
            else: #L or R no looping
                reward = 0.5 / float(center_detour+1)
        else:
            self.reward = -1

        return self.ob, self.reward, self.done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        # print("New episode")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Reset the step iterator
        self.iterator = 0

        if self.reset_position is True:
            # rospy.wait_for_service('/gazebo/reset_simulation')
            rospy.wait_for_service('/gazebo/set_model_state')
            new_initial_pose = self.getRandomPosition()
            try:
                self.set_position_proxy(new_initial_pose)
                # self.reset_proxy()
            except (rospy.ServiceException) as e:
                print ("/gazebo/reset_simulation service call failed")
                print("/gazebo/set_model_state service call failed")


        # Take an observation
        self.ob = None
        while(self.ob is None):
            # print("  --> Acquiring first observation")
            self.ob = self.take_observation()

        return self.ob

    def getRandomPosition(self):
        random_pose = ModelState()

        tmp_x = np.random.uniform( low=self.min_x, high=self.max_x)
        tmp_y = np.random.uniform( low=self.min_y, high=self.max_y)
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

        random_pose.reference_frame = 'world'
        random_pose.model_name = 'mobile_base'

        return random_pose