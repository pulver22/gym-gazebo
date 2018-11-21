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
        self._drl_sub = rospy.Subscriber('/drl/camera', numpy_msg(HeaderString), self.observation_callback)
        # self._sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.observation_callback)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_position_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self._observation_msg = None
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

        # Observation space
        self.observation_high = 255
        self.observation_low = 0
        self.observation_space = spaces.Box(self.observation_low, self.observation_high, shape=(self.img_rows, self.img_cols, self.img_channels), dtype=np.uint8)

        # Environment hyperparameters
        self.min_x = -5.0
        self.max_x = - self.min_x
        self.min_y = - 5.0
        self.max_y = - self.min_y

        self.reward_range = (-np.inf, np.inf)


        self._seed()

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message.data
        print("Callback")
        print(self._observation_msg)

    def take_observation(self):
        """
        Take observation from the environment and return itself.
        """
        # min_range = 0.2
        # done = False
        # obs_message = self._observation_msg
        # print("Obs_msgs: ", self._observation_msg)
        obs_message = None
        bridge = CvBridge()
        print("Camera Empty: ", obs_message)
        while obs_message is None:
            try:
                obs_message = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                # obs_message = rospy.wait_for_message('/drl/camera', HeaderString)

                # Convert from sensor_msgs::Image to cv::Mat
        #         cv_image = bridge.imgmsg_to_cv2(obs_message, desired_encoding="passthrough")
        #         # Access global variable and store image as numpy.array
        #         obs_message = np.asarray(cv_image)
            except:# CvBridgeError as ex:
                print ("ERROR!!")#, ex)
        print("Data:", obs_message.data)
        # print("Data: ", np.fromstring(obs_message.data, dtype=int))
        # print("Type:", type(obs_message))
        # print("Type data:", type(obs_message.data))
        # print("Camera: ", string(obs_message.data))
        # image_data = obs_message.data
        # print("Camera: ", obs_message)
        # bridge = CvBridge()
        # try:
        #     # Convert from sensor_msgs::Image to cv::Mat
        #     cv_image = bridge.imgmsg_to_cv2(obs_message, desired_encoding="passthrough")
        #     # Access global variable and store image as numpy.array
        #     obs_message = np.asarray(cv_image)
        # except CvBridgeError as ex:
        #     print ("ERROR!!", ex)

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
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
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

        #TODO: Create an action message
        self.vel_pub.publish(self.get_velocity_message(action))
        # print("[", self.iterator, "] Action selected [lin, ang]: ", action)

        self.ob = self.take_observation()


        #########################
        ##         DONE        ##
        #########################
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        self.done = self.calculate_done(data)

        #########################
        ##         STATE       ##
        #########################
        success = False
        cv_image = None
        while(self.ob is None or success is False):
            try:
                self.ob = self.take_observation()
                h = self.ob.height
                w = self.ob.width
                cv_image = CvBridge().imgmsg_to_cv2(self.ob, "bgr8")
                # temporal fix, check image is not corrupted
                if not (cv_image[h // 2, w // 2, 0] == 178 and cv_image[h // 2, w // 2, 1] == 178 and cv_image[
                    h // 2, w // 2, 2] == 178):
                    success = True
                else:
                    # pass
                    print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        self.ob = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])

        #########################
        ##        REWARD       ##
        #########################
        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)
        action_sum = sum(self.last50actions)

        # Add center of the track reward
        # len(data.ranges) = 100
        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len - (laser_len / 5):laser_len - (laser_len / 10)])  # 80-90
        right_sum = sum(data.ranges[(laser_len / 10):(laser_len / 5)])  # 10-20
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

        return self.ob, self.reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        #
        # # Unpause simulation to make observation
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")
        # #read laser data
        # data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        #     except:
        #         pass
        #
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")
        #
        # state,done = self.calculate_observation(data)

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

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        # Take an observation
        self.ob = None
        while(self.ob is None):
            self.ob = self.take_observation()
        #print("Observation after reset: ", self.ob)

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