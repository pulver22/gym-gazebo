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
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from rosgraph_msgs.msg import Clock
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats, HeaderString

from cv_bridge import CvBridge, CvBridgeError

#from tf.transformations import quaternion_from_euler




class GazeboThorvaldCameraCnnPPOEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboThorvald.launch")
        self.vel_pub = rospy.Publisher('nav_vel', Twist, queue_size=5)
        # self._drl_sub = rospy.Subscriber('/drl/camera', numpy_msg(HeaderString), self.observation_callback)
        self.camera_sub = rospy.Subscriber('/thorvald_ii/kinect2/hd/image_color_rect', Image, self.observation_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clock_callback)
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
        self.max_episode_steps = 100  # limit the max episode step
        self.iterator = 0  # class variable that iterates to accounts for number of steps per episode
        self.reset_position = True
        self.use_euler_angles = False

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
        self.observation_high = 255
        self.observation_low = 0
        # self.observation_space = spaces.Box(self.observation_low, self.observation_high, shape=(self.img_rows, self.img_cols, self.img_channels), dtype=np.uint8)  # Without goal info
        self.goal_info = np.zeros(shape=(self.img_rows, 1, 1))  # Arrays need to have same dimesion in order to be concatened
        self.observation_space = spaces.Box(self.observation_low, self.observation_high, shape=(self.img_rows, self.img_cols + 1, self.img_channels), dtype=np.float16)  # With goal info


        # Environment hyperparameters
        self.initial_pose = None
        self.min_x = -5.0
        self.max_x = 5.0
        self.min_y = - -1.5
        self.max_y = 1.5
        self.offset = 3.0
        self.target_position = [None, None]
        self.model_name = 'thorvald_ii'
        self.reference_frame = 'world'
        self.last_clock_msg = None
        self.last_step_ts = None
        self.skip_time = 500000000  # expressed in nseconds



        self.reward_range = (-1000.0, 1000)
        self.penalization = - 200
        self.positive_reward = 800
        self.acceptance_distance = 0.20
        self.proximity_distance = 0.5
        self.distance = None
        self.robot_abs_pose = None
        self.euler_bearing = None

        self.time_start = 0.0
        self.time_stop = 0.0
        self.rospy_time_start = 0.0
        self.rospy_time_stop = 0.0

        self.r = rospy.Rate(30)

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
        # print("Wallclock2: ", rospy.rostime.is_wallclock())
        # print("Rospyclock2: ", rospy.rostime.get_rostime().secs )
        self.iterator += 1
        print("[B]Time difference between step: ", (float(time.time()) - self.time_stop), " sec")
        print("[B]ROSPY Time difference between step: ", abs(rospy.get_rostime().nsecs - self.rospy_time_stop)*1e-9, " sec")
        # print("[B]ROSPY Time ", rospy.get_time())

        self.time_stop = float(time.time())
        self.rospy_time_stop = float(rospy.get_rostime().nsecs)

        # print("Time difference between step: ", (self.time_stop - self.time_start), " sec")
        # print("ROSPY Time difference between step: ", (self.rospy_time_stop - self.rospy_time_start) * 1e-9, " sec")


        #########################
        ##       ACTION        ##
        #########################
        # Unpause simulation
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        #     print("UnPausing")
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/unpause_physics service call failed")

        # print("  --> Sending action")
        # TODO: Create an action message
        # print("C: ", self.last_clock_msg)
        # print("TS: ", self.last_step_ts)
        # before_action_ts = self.last_clock_msg
        # action_time = self.last_step_ts - before_action_ts
        # action_iterator = 1
        # timer = abs(rospy.rostime.get_rostime().nsecs)
        # timer = self.last_clock_msg
        # print("timer: ", timer)
        # delta = abs( abs(rospy.Time.now().nsecs) - timer)
        # old_clock = self.last_clock_msg
        # print("Oldclock: ", old_clock)

        # while (abs(rospy.rostime.get_rostime().nsecs - timer) < self.skip_time):
        #     pass
            # print("Actualclocl: ", rospy.get_rostime().nsecs)
            # print("d: ", delta)
            # print("AI: ", action_iterator)
        self.vel_pub.publish(self.get_velocity_message(action))
        rospy.sleep(rospy.Duration(0, self.skip_time))
            # self.last_step_ts = self.last_clock_msg
            # print("--> ", self.last_step_ts)
            # action_time = self.last_step_ts - before_action_ts
            # action_iterator = action_iterator + 1
            # print("timer2: ", rospy.get_rostime().nsecs)
            # delta = abs( abs(rospy.Time.now().nsecs) - timer)
        # print("Action sent")
        # wait until action gets executed
        # rospy.sleep(rospy.Duration(1.0))  # sleep for half second
        # print("[", self.iterator, "] Action selected [lin, ang]: ", action)

        # Pause simulation
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     self.pause()
        #     print("Pausing")
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")






        #########################
        ##         DONE        ##
        #########################
        # TODO: check that done is set to True when walls are within safety distance [it does seems not to work]
        # TODO: collision check must be performed using the physic engine
        # laser_ranges = None
        # while laser_ranges is None:
        #     try:
        #         # print("  --> Acquiring laser data")
        #         laser_ranges = self._lidar_msg
        #     except:
        #         rospy.logerr("Error while reading Lidar")
        #
        # self.done, self.reward = self.calculate_collision(laser_ranges)



        #########################
        ##        REWARD       ##
        #########################

        # print("  --> Calculating reward")
        # Calculate the reward only if there is no collision
        if self.done == False:
            # Calculate actual distance from robot
            self.getGoalDistance()
            # Calculate reward
            # self.reward = self.reward + self.getReward()  # NOTE: to use when checking collision in "DONE" block
            self.reward = self.getReward()
        #print("Reward: ", self.reward)
        # If the reward is greater than zero, we are in proximity of the goal. So Done needs to be set to true
        if self.reward >= 0 or self.iterator == (self.max_episode_steps -1):
            self.done = True


        # Printing info
        if self.done == True:
            # rospy.logwarn("Done: " + str(self.done))
            print("Final distance to goal: ", self.distance)
            if self.reward >= 0:
                rospy.logwarn("The robot reached its target")
            else:
                pass

        #########################
        ##         STATE       ##
        #########################
        self.ob = None
        while (self.ob is None):
            try:
                # print("  --> Acquiring observation")
                self.ob = self.take_observation()
            except:
                rospy.logerr("Problems acquiring the observation")

        self.goal_info[0] = self.distance
        if self.use_euler_angles == True:
            self.getBearingEuler()
            self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        else:
            self.goal_info[1] = self.robot_abs_pose.pose.orientation.x
            self.goal_info[2] = self.robot_abs_pose.pose.orientation.y
            self.goal_info[3] = self.robot_abs_pose.pose.orientation.z
            self.goal_info[4] = self.robot_abs_pose.pose.orientation.w

        # Append the goal information (distance and bearing) to the observation space
        self.ob = np.append(self.ob, self.goal_info, axis=1)


        # self.rospy_time_start = float(rospy.get_rostime().nsecs)
        # self.time_start = float(time.time())
        # print("[A]Time difference between step: ", (float(time.time()) - self.time_stop), " sec")
        # print("[A]ROSPY Time difference between step: ", (float(rospy.get_time()) - self.rospy_time_stop), " sec")
        # self.rospy_time_stop = float(rospy.get_time())
        # self.time_stop = float(time.time())
        # r.sleep()
        # break
        return self.ob, self.reward, self.done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        print("New episode")

        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # Reset the step iterator
        self.iterator = 0

        # Unpause simulation
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        #     print("UnPausing")
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/unpause_physics service call failed")

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

        # Reset the target position and calculate distance robot-target
        self.getRandomTargetPosition()
        self.getGoalDistance()



        # Take an observation
        self.ob = None
        while (self.ob is None):
            try:
                # print("  --> Acquiring observation")
                self.ob = self.take_observation()
            except:
                rospy.logerr("Problems acquiring the observation")

        self.goal_info[0] = self.distance
        if self.use_euler_angles == True:
            self.getBearingEuler()
            self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        else:
            self.goal_info[1] = self.robot_abs_pose.pose.orientation.x
            self.goal_info[2] = self.robot_abs_pose.pose.orientation.y
            self.goal_info[3] = self.robot_abs_pose.pose.orientation.z
            self.goal_info[4] = self.robot_abs_pose.pose.orientation.w

        # Append the goal information (distance and bearing) to the observation space
        self.ob = np.append(self.ob, self.goal_info, axis=1)

        # Pause the simulation
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     self.pause()
        #     print("Pausing")
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")


        # Printing info
        print("New starting position: [", self.initial_pose.pose.position.x, ",", self.initial_pose.pose.position.y, ", ",
              self.initial_pose.pose.position.z, "]")
        print("New target: ", self.target_position)
        print("Distance to goal: ", self.distance)

        self.last_step_ts = self.last_clock_msg
        self.rospy_time_stop = float(rospy.Time.now().nsecs)
        self.done = False

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

        random_pose.reference_frame = self.reference_frame
        random_pose.model_name = self.model_name

        return random_pose

    def getBearingEuler(self):
        """
        Get the robot absolute bearing to the goalself.
        """
        q = np.quaternion(self.robot_abs_pose.pose.orientation.x,
                                            self.robot_abs_pose.pose.orientation.y,
                                            self.robot_abs_pose.pose.orientation.z,
                                            self.robot_abs_pose.pose.orientation.w)  # arg is expressed in quaternion
        self.euler_bearing = quaternion.as_euler_angles(q)
        # print("Quaternion: ", str(self.robot_abs_pose.pose.orientation))
        # print("Euler: ", str(self.euler_bearing))

    def getRandomTargetPosition(self):
        """
        Generate a random target within the arena
        :return:
        """
        low = self.initial_pose.pose.position.x - self.offset
        high = self.initial_pose.pose.position.x + self.offset
        target_x = np.random.uniform(low=low, high=high)
        low = self.initial_pose.pose.position.y - self.offset
        high = self.initial_pose.pose.position.y + self.offset
        target_y = np.random.uniform(low=low, high=high)
        # target_x = np.random.uniform(low=self.min_x, high=self.max_x)
        # target_y = np.random.uniform(low=self.min_y, high=self.max_y)
        target_x = 0.0
        target_y = 0.0
        target_z = 0.0
        self.target_position = np.array((target_x, target_y, target_z))

    def getGoalDistance(self):
        """
        Calculate the distance between two points in space
        :param goal_a:
        :param goal_b:
        :return:
        """
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.robot_abs_pose = model_coordinates(self.model_name, self.reference_frame)
        position = np.array(
            (self.robot_abs_pose.pose.position.x, self.robot_abs_pose.pose.position.y, self.robot_abs_pose.pose.position.z))

        assert position.shape == self.target_position.shape
        # print("goal_distance", np.linalg.norm(goal_a - goal_b, axis=-1))
        self.distance = np.linalg.norm(position - self.target_position, axis=-1)

    def getReward(self):
        """
        Calculate the reward as euclidean distance from robot to target
        :return:
        """

        #if self.distance < self.proximity_distance:
        if self.distance < self.acceptance_distance:
            return self.positive_reward
         #   else:
          #      return self.positive_reward * 0.01 - self.distance
        else:
            return - self.distance.astype(np.float32) * 0.1
