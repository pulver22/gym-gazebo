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
from gazebo_msgs.msg import ModelState, ContactState, ModelStates
from rosgraph_msgs.msg import Clock
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats, HeaderString

from cv_bridge import CvBridge, CvBridgeError

#from tf.transformations import quaternion_from_euler




class GazeboThorvaldCameraCnnPPOEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboThorvald.launch", "//home/pulver/ncnr_ws/src/gazebo-contactMonitor/launch/contactMonitor.launch")
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

        ##########################
        ##      PARAMETERS      ##
        ##########################
        self.max_episode_steps = 200  # limit the max episode step
        self.reward_range = (-1000.0, 200)
        self.penalization = - 200
        self.positive_reward = 200
        self.acceptance_distance = 1.0
        self.proximity_distance = 1.5
        self.min_x = -4.0
        self.max_x = 4.0
        self.min_y = - 4.0
        self.max_y = 4.0
        self.offset = 3.0
        self.max_distance = 15.0
        self.skip_time = 500000000  # expressed in nseconds
        self.model_name = 'thorvald_ii'
        self.reference_frame = 'world'
        ##########################

        self._observation_msg = None
        self._lidar_msg = None
        self._last_obs_header = None
        self._last_lidar_header = None
        self.obs = None
        self.reward = 0
        self.done = False
        self.iterator = 0  # class variable that iterates to accounts for number of steps per episode
        self.reset_position = True
        self.use_cosine_sine = True
        self.fake_images = False
        self.collision_detection = True

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
        self.target_position = [None, None]
        self.distance = None
        self.robot_abs_pose = None
        self.robot_rel_orientation = None
        self.robot_target_abs_angle = None
        self.euler_bearing = None
        self.last_collision = None
        self.pose_acceptable = False

        self.last_clock_msg = None
        self.last_step_ts = None
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

    def calculate_collision_lidar(self,data):
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
        # print("[B]Time difference between step: ", (float(time.time()) - self.time_stop), " sec")
        # print("[B]ROSPY Time difference between step: ", abs(rospy.get_rostime().nsecs - self.rospy_time_stop)*1e-9, " sec")
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
        #     # print("UnPausing")
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
        #     # print("Pausing")
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")


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
            if self.collision_detection == True:
                self.reward += self.getCollisionPenalty()
        #print("Reward: ", self.reward)

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
        # self.done, self.reward = self.calculate_collision_lidar(laser_ranges)

        # If the reward is greater than zero, we are in proximity of the goal. So Done needs to be set to true
        if self.reward >= 0 or self.iterator == (self.max_episode_steps -1) or self.reward <= -10.0:
            self.done = True
            self.pose_acceptable = False


        #########################
        ##         STATE       ##
        #########################
        self.ob = None
        while (self.ob is None):
            try:
                # print("  --> Acquiring observation")
                if self.fake_images == True:
                    self.ob = np.zeros(shape=(84, 84, 1))  # DEBUG
                else:
                    self.ob = self.take_observation()
            except:
                rospy.logerr("Problems acquiring the observation")

        self.goal_info[0] = self.normalise(value=self.distance, min=0.0, max=self.max_distance)

        # print("Distance: ", str(self.goal_info[0]))

        self.getBearingEuler()
        if (0 < abs(self.euler_bearing[0]) < 0.52) or (2.62 < abs(self.euler_bearing[0]) < 3.14):
            rospy.logerr("The robot is flipped")
            self.done = True
            self.pose_acceptable = False
            self.reward = self.penalization
        # self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        self.getRobotTargetAbsAngle()
        self.getRobotRelOrientation()
        self.goal_info[1] = self.robot_rel_orientation
        self.goal_info[1] = self.normalise(value=self.goal_info[1], min=-180.0, max=180.0)
        if self.use_cosine_sine == True:
            self.goal_info[1] = math.cos(self.robot_rel_orientation * 3.14 / 180.0) # angles must be expressed in radiants
            self.goal_info[2] = math.sin(self.robot_rel_orientation * 3.14 / 180.0)
            # Normalise the sine and cosine
            self.goal_info[1] = self.normalise(value=self.goal_info[1], min=-1.0, max=1.0)
            self.goal_info[2] = self.normalise(value=self.goal_info[2], min=-1.0, max=1.0)
            # print("Bearing: ", self.robot_rel_orientation, ",",str(self.goal_info[1]), ", ", str(self.goal_info[2]))

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
        # Printing info
        if self.done == True:
            # rospy.logwarn("Done: " + str(self.done))
            print("Final distance to goal: ", self.distance)
            if self.reward >= 0:
                rospy.logwarn(" -> The robot reached its target.")
            elif self.reward <= -10.0:
                rospy.logerr("  -> The robot crashed into a static obstacle.")
            else:
                pass

        return self.ob, self.reward, self.done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        msg = None
        while msg == None:
            msg = rospy.wait_for_message("/fag", ContactState)
            print("Waiting")

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
        #     # print("UnPausing")
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/unpause_physics service call failed")

        # Reset robot position
        if self.reset_position is True:
            # rospy.wait_for_service('/gazebo/reset_simulation')
            rospy.wait_for_service('/gazebo/set_model_state')
            print("Waiting for object list")
            while self.pose_acceptable == False:
                self.initial_pose = self.getRandomRobotPosition()
                self.pose_acceptable = self.checkRobotPose()

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
                if self.fake_images == True:
                    self.ob = np.zeros(shape=(84, 84, 1))  # DEBUG
                else:
                    self.ob = self.take_observation()
            except:
                rospy.logerr("Problems acquiring the observation")

        self.goal_info[0] = self.distance

        self.getBearingEuler()
        # self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        self.getRobotTargetAbsAngle()
        self.getRobotRelOrientation()
        self.goal_info[1] = self.robot_rel_orientation
        if self.use_cosine_sine == True:
            self.goal_info[1] = math.cos(self.robot_rel_orientation * 3.14 / 180.0)  # angles must be expressed in radiants
            self.goal_info[2] = math.sin(self.robot_rel_orientation * 3.14 / 180.0)
            # Normalise the sine and cosine
            self.goal_info[1] = self.normalise(value=self.goal_info[1], min=-1.0, max=1.0)
            self.goal_info[2] = self.normalise(value=self.goal_info[2], min=-1.0, max=1.0)


        # Append the goal information (distance and bearing) to the observation space
        self.ob = np.append(self.ob, self.goal_info, axis=1)

        # Pause the simulation
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     self.pause()
        #     # print("Pausing")
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

    def getRandomRobotPosition(self):
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
        x = self.robot_abs_pose.pose.orientation.x
        y = self.robot_abs_pose.pose.orientation.y
        z = self.robot_abs_pose.pose.orientation.z
        w = self.robot_abs_pose.pose.orientation.w

        q = np.quaternion(x, y, z, w)  # arg is expressed in quaternion
        self.euler_bearing = quaternion.as_euler_angles(q)
        # FIX: self.euler_beraing[1] is in [0,pi] while self.euler_bearing[0] is in [-pi, pi] and gives the sign
        if z * w < 0:
            self.euler_bearing[1] = 2 * math.pi - self.euler_bearing[1]  # cast to [0, 2pi]
        # print("Quaternion: ", str(self.robot_abs_pose.pose.orientation))
        # print(" -> Euler: ", str(self.euler_bearing))

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

    def getRobotTargetAbsAngle(self):
        """
        Calculate the angle (in degreed) between the X-axis and the vector connecting the robot to its target
        :return:
        """
        delta_y = self.robot_abs_pose.pose.position.y - self.target_position[1]
        delta_x = self.robot_abs_pose.pose.position.x - self.target_position[0]
        # cos_B = delta_x / (math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2)))
        # sin_B = delta_y / (math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2)))
        self.robot_target_abs_angle = math.atan2(delta_y, delta_x)
        # print("Delta_x: ", str(delta_x), " delta_y: ", str(delta_y))
        # print("[1]", str(self.robot_target_abs_angle))
        if delta_y <= 0:
            self.robot_target_abs_angle = self.robot_target_abs_angle + 2 * math.pi
        # print("[2]", str(self.robot_target_abs_angle))
        self.robot_target_abs_angle = self.robot_target_abs_angle * 180 / math.pi
        # print("[3]", str(self.robot_target_abs_angle))

    def getRobotRelOrientation(self):
        """
        Get the relative angle between the robot orientation and the vector connecting the robot to its target
        :return:
        """
        sign = -1

        self.robot_rel_orientation = abs(self.robot_target_abs_angle - (self.euler_bearing[1] * 180 / 3.14))
        # if self.robot_target_abs_angle >= 180:
        # Make the If statement more readable

        alpha = self.euler_bearing[1] * 180 / 3.14
        beta = self.robot_target_abs_angle
        # print("Alpha: ", str(alpha))
        # print("Beta: ", str(beta))

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
                ( (180.0 <= beta <= 270.0) and (beta <= alpha <= 360.0) or (0.0 <= alpha <= (beta - 180.0)) ) or \
                ((270.0 <= beta <=  360.0) and (-(360.0 - beta) <= alpha <= 90.0 + (360.0 - beta))):
            sign = 1
        #   rospy.logerr("Signed changed!")
        # print("No sign changed")
        self.robot_rel_orientation = sign * abs(180 -  self.robot_rel_orientation)
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

    def getCollisionPenalty(self):
        """
        Check for collision and penalise them
        :return:
        """
        penalty = 0.0
        if self.last_collision != None:
            for wrench in self.last_collision.wrenches:
                penalty += min(0.01 * math.sqrt(pow(wrench.force.x,2) + pow(wrench.force.y,2) + pow(wrench.force.z,2)), 10)

        # if penalty > 0.0:
        #     rospy.logerr("Collision detected, penalty = %s", penalty)
        # Reset last collision to prevent to use old data
        self.last_collision = None
        return -penalty

    def checkRobotPose(self):
        """
        Check if the robot is within an object
        :return:
        """
        objects_msg = None
        while objects_msg == None:
            objects_msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)


        for i in range(0, len(objects_msg.name)):
            # print("Checking pose ", (i+1), "of", len(objects_msg.name))
            if (objects_msg.pose[i].position.x - self.proximity_distance < self.initial_pose.pose.position.x < objects_msg.pose[i].position.x + self.proximity_distance):
                if (objects_msg.pose[i].position.y - self.proximity_distance < self.initial_pose.pose.position.y < objects_msg.pose[i].position.y + self.proximity_distance):
                    return False

        return True

