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
from gazebo_msgs.msg import ModelStates, ContactState
from rosgraph_msgs.msg import Clock
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats, HeaderString

from cv_bridge import CvBridge, CvBridgeError

from gym_gazebo.envs.thorvald.navigation_utilities import NavigationUtilities




class GazeboThorvaldCameraCnnPPOEnvSlim(gazebo_env.GazeboEnv):

    def __init__(self):


        ##########################
        ##      PARAMETERS      ##
        ##########################
        self.max_episode_steps = 200  # limit the max episode step
        self.reward_range = (-1000.0, 200)
        self.penalization = - 200
        self.positive_reward = 200
        self.tolerance_penalty = -2.0
        self.acceptance_distance = 1.0
        self.proximity_distance = 3.0
        self.world_xy = [-3.0, 3.0, -3.0, 3.0]
        # self.world_y = [-6.0, 6.0]
        self.robot_xy = [-4.0, 4.0, -4.0, 4.0]
        # self.robot_y = [-3.0, 3.0]
        self.offset = 3.0
        self.max_distance = 15.0
        self.skip_time = 500000000  # expressed in nseconds
        self.navigation_multiplyer = 100.0
        self.model_name = 'thorvald_ii'
        self.reference_frame = 'world'
        self.use_cosine_sine = True
        self.fake_images = False
        self.collision_detection = True
        self.synch_mode = True #TODO: if set to True, the code doesn't continue because ROS is synch with gazebo
        self.reset_position = True
        self.use_depth = False
        self.registered = False
        self.use_combined_depth = False
        self.use_stack_memory = False
        self.episodes_reset = 15
        # Camera setting
        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1
        if self.use_combined_depth is True:
            self.img_channels += 1
        self.obs = np.zeros(shape=(self.img_rows, self.img_cols + 1, self.img_channels))
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboThorvald.launch", self.collision_detection,
                                      "/home/pulver/ncnr_ws/src/gazebo-contactMonitor/launch/contactMonitor.launch")

        print("==== CONFIGURATION ====")
        print("use_cosine_sine: ", self.use_cosine_sine)
        print("fake_images: ", self.fake_images)
        print("collision_detection: ", self.collision_detection)
        print("sych_mode: ", self.synch_mode)
        print("reset_position: ", self.reset_position)
        print("use_depth: ", self.use_depth)
        print("use_combined_depth: ", self.use_combined_depth)
        print("use_stack_memory: ", self.use_stack_memory)
        print("Observation shape: ", np.shape(self.obs))
        print("=======================")


        if (self.use_stack_memory is True and self.use_depth is True):
            raise Exception('If using a stack of images, depth is not supported yet')
        # assert (self.use_stack_memory is True and self.use_depth is False) or \
        #        (self.use_stack_memory is False and self.use_depth is True), "If using a stack of images, depth is not supported yet"
        ##########################

        self._observation_msg = None
        self._depth_msg = None
        self._lidar_msg = None
        self._last_obs_header = None
        self._last_depth_header = None
        self._last_lidar_header = None
        self._last_contact_header = None
        # Lidar setting
        self.min_range = 0.5
        # Goal_info needs to have same dimension  of images in order to be concatenated
        self.goal_info = np.zeros(shape=(self.img_rows, 1, 1))
        self.reward = 0
        self.done = False
        self.iterator = 0  # class variable that iterates to accounts for number of steps per episode
        self.episodes_counter = 0
        self.steps_counter = 0

        ##########################
        ##     ACTION SPACE     ##
        ##########################
        # self.velocity_low = np.array([-0.3, -0.3, -0.2], dtype=np.float32)
        # self.velocity_high = np.array([0.3, 0.3, 0.2], dtype=np.float32)
        self.velocity_low = np.array([0, -0.2], dtype=np.float32)
        self.velocity_high = np.array([0.3, 0.2], dtype=np.float32)
        self.action_space = spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32)


        ##########################
        ##       OBS SPACE      ##
        ##########################
        self.bridge = CvBridge()
        self.observation_high = 1.0
        self.observation_low = 0.0
        # self.observation_high = 1.0  # DEBUG: mockup images
        # self.observation_low = 0.0  # DEBUG: mockup images
        # self.observation_space = spaces.Box(self.observation_low, self.observation_high, shape=(self.img_rows, self.img_cols, self.img_channels), dtype=np.uint8)  # Without goal info
        self.observation_space = spaces.Box(low=self.observation_low,
                                            high=self.observation_high,
                                            shape=(self.img_rows, self.img_cols + 1, self.img_channels),
                                            dtype=np.float16)  # With goal info

        ##########################
        ##     ENVIRONMENT      ##
        ##########################
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
        self.objects_list = None
        self.objects_pose = []
        self._seed()

        self.nav_utils = NavigationUtilities(reference_frame=self.reference_frame, model_name=self.model_name,
                                             proximity_distance=self.proximity_distance, acceptance_distance=self.acceptance_distance,
                                             offset=self.offset, positive_reward=self.positive_reward)

        ##########################
        ##          ROS         ##
        ##########################
        self.vel_pub = rospy.Publisher('nav_vel', Twist, queue_size=5)
        self.camera_sub = rospy.Subscriber('/thorvald_ii/kinect2/hd/image_color_rect', Image, self.observationCallback)
        if self.registered is True:
            self.depth_sub = rospy.Subscriber('depth_registered/image_rect', Image, self.depthCallback )  # registered depth
        else:
            self.depth_sub = rospy.Subscriber('/thorvald_ii/kinect2/sd/image_depth_rect', Image, self.depthCallback)
        # self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clockCallback)
        # self.objects_sub = rospy.Subscriber('/gazebo/model_states', ModelState, self.objects_callback)
        self.collision_sub = rospy.Subscriber('/collision_data', ContactState, self.contactCallback)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_position_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # Gazebo specific services to start/stop its behavior and facilitate the overall RL environment
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.time_start = 0.0
        self.time_stop = 0.0
        self.rospy_time_start = 0.0
        self.rospy_time_stop = 0.0
        self.r = rospy.Rate(20)


    # def objects_callback(self, message):
    #     """
    #     Callback for retrieving the list of objects present in the world
    #     """
    #     self.objects_list = message.name


    def clockCallback(self, message):
        """
        Callback method for the subscriber of the clock topic
        :param message:
        :return:
        """
        # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
        self.last_clock_msg = int(message.clock.nsecs)
        # print(self.last_clock_msg)

    def observationCallback(self, message):
        """
        Callback method for the subscriber of the camera
        """

        # print("Image received")
        if message.header.seq != self._last_obs_header:
            self._last_obs_header = message.header.seq
            self._observation_msg = message
        else:
            rospy.logerr("Not receiving images")


    def depthCallback(self, message):  # TODO still too noisy!
        """
        Callback method for the subscriber of the depth camera
        """
        # print("Depth callback!")
        if message.header.seq != self._last_depth_header:
            self._last_depth_header = message.header.seq
            self._depth_msg = message
            # print(self._depth_msg.header.seq)
        else:
            rospy.logerr("Not receiving images")


    def takeDepth(self):
        """
        Take depth observation from the environment and return itself.
        """
        # print("Takedepth!")
        depth_message = None
        while depth_message is None:
            try:
                depth_message = self._depth_msg
                # print("depth_message: ", depth_message)
            except:  # CvBridgeError as ex:
                rospy.logerr("ERROR!!")  # , ex)
        # The depth image is a single-channel float32 image
        # the values is the distance in mm in z axis
        cv_image = self.bridge.imgmsg_to_cv2(depth_message, "passthrough")
        # Crop the image to be 1080*1080, keeping the same centre of the original one
        if self.registered is True:
            cv_image = cv_image[:, 419:1499]
        else:
            #  The original resolution is 424*512, crop it to 424*424
            cv_image = cv_image[:, 43:467]
        # cv_image[np.isnan(cv_image)] = 0 # TODO: faster method (10x) than np.nan_to_num which does not work
        cv_image = np.nan_to_num(cv_image)
        # cv2.imwrite('/home/pulver/Desktop/img_original.png', cv_image)
        cv_image_norm = cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        cv_image_norm = cv_image_norm / 255.0
        cv2.imwrite('/home/pulver/Desktop/img_norm.png', cv_image_norm)
        # Resize to the desired size
        cv_image = cv2.resize(cv_image_norm, (self.img_rows, self.img_cols), interpolation=cv2.INTER_CUBIC)
        depth_message =  cv_image.reshape(cv_image.shape[0], cv_image.shape[1], 1)
        return depth_message

    def takeObservation(self):
        """
        Take observation from the environment and return itself.
        """
        obs_message = None
        # print("Camera Empty: ", obs_message)
        while obs_message is None:
            try:
                obs_message = self._observation_msg
            except:# CvBridgeError as ex:
                rospy.logerr ("ERROR!!")#, ex)
        # Convert from sensor_msgs::Image to cv::Mat
        cv_image = self.bridge.imgmsg_to_cv2(obs_message, desired_encoding="bgr8")
        # Convert the image to grayscale
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Crop the image to be 1080*1080, keeping the same centre of the original one
        cv_image = cv_image[:, 419:1499]
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        cv_image_norm = cv_image / 255.0
        # cv_image_norm = cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Resize and reshape the image according to the network input size
        cv_image = cv2.resize(cv_image_norm, (self.img_rows, self.img_cols), interpolation=cv2.INTER_CUBIC)
        obs_message = cv_image.reshape( cv_image.shape[0], cv_image.shape[1], 1)
        return obs_message


    def lidarCallback(self, message):
        """
        Callback method for the subscriber of lidar
        """
        if message.header.seq != self._last_lidar_header:
            self._last_lidar_header = message.header.seq
            self._lidar_msg =  np.array(message.ranges)
        else:
            rospy.logerr("Not receiving lidar readings")

    def contactCallback(self, message):
        """
        Parse ContactState messages for possible collision (filter those involving the ground)
        :param message:
        :return:
        """
        if "ground_plane" in message.collision1_name or "ground_plane" in message.collision2_name:
            pass
        elif "thorvald_ii" in message.collision1_name or "thorvald_ii" in message.collision2_name:
            self.last_collision = message


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


        self.time_start = float(time.time())
        self.rospy_time_start = rospy.get_rostime()


        #########################
        ##       ACTION        ##
        #########################
        # print("[", self.iterator, "]Action: ", action)
        # action = np.clip(action, self.action_space.low,
        #                  self.action_space.high)
        # print("     [", self.iterator, "]Action: ", action)
        # print("Before unpausing")
        if self.synch_mode == True:
            # Unpause simulation
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
                # print("UnPausing")
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

        # start = rospy.get_rostime()
        self.vel_pub.publish(self.nav_utils.getVelocityMessage(action))
        rospy.sleep(rospy.Duration(0, self.skip_time))
        # stop = rospy.get_rostime()
        # print("SRostime: ", abs(stop.secs - start.secs))
        # print("NRostime: ", abs(stop.nsecs - start.nsecs))

        if self.synch_mode == True:
            # Pause simulation
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                self.pause()
                # print("Pausing")
            except (rospy.ServiceException) as e:
                print("/gazebo/pause_physics service call failed")
        # rospy.sleep(rospy.Duration(0, self.skip_time))
        #########################
        ##         STATE       ##
        #########################
        last_ob = None
        last_depth = None
        while (last_ob is None and last_depth is None):
            try:
                if self.fake_images == True:
                    last_ob = np.zeros(shape=(84, 84, 1))  # DEBUG
                else:
                    last_ob = self.takeObservation()
                    last_depth = self.takeDepth()
                    # self.takeDepth()
                    # last_ob = self.observation
                    # print("[Image] min: {}, max: {}".format(np.min(last_ob), np.max(last_ob)))
            except:
                rospy.logerr("Problems acquiring the observation")

        # Calculate actual distance from robot
        self.robot_abs_pose = self.nav_utils.getRobotAbsPose()
        self.distance = self.nav_utils.getGoalDistance(self.robot_abs_pose, self.target_position)
        self.goal_info[0] = self.nav_utils.normalise(value=self.distance, min=0.0, max=self.max_distance)

        # Calculate the relative orientation of the robot to the goal
        self.euler_bearing = self.nav_utils.getBearingEuler(self.robot_abs_pose)
        if (0 < abs(self.euler_bearing[0]) < 0.52) or (2.62 < abs(self.euler_bearing[0]) < 3.14):
            rospy.logerr("The robot is flipped")
            self.done = True
            self.pose_acceptable = False
            self.reward = self.penalization
        # self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        self.robot_target_abs_angle = self.nav_utils.getRobotTargetAbsAngle(self.robot_abs_pose, self.target_position)
        # self.robot_rel_orientation = np.math.radians(self.nav_utils.getRobotRelOrientation(self.robot_target_abs_angle, self.euler_bearing[1]))
        self.robot_rel_orientation = self.nav_utils.getRobotRelOrientationAtan2(self.robot_target_abs_angle,
                                                                                self.euler_bearing[1])
        self.goal_info[1] = self.robot_rel_orientation
        # print("[", self.iterator, "]Angle: ", np.degrees(self.goal_info[1]))
        # self.goal_info[1] = self.nav_utils.normalise(value=self.goal_info[1], min=-180.0, max=180.0)
        if self.use_cosine_sine == True:
            self.goal_info[1] = math.cos(self.robot_rel_orientation)  # angles must be expressed in radiants
            self.goal_info[2] = math.sin(self.robot_rel_orientation)
            # print("     [distance, cosine, sine]: ", self.goal_info[0,:,:], self.goal_info[1,:,:], self.goal_info[2,:,:])
            # Normalise the sine and cosine
            # self.goal_info[1] = self.nav_utils.normalise(value=self.goal_info[1], min=-1.0, max=1.0)
            # self.goal_info[2] = self.nav_utils.normalise(value=self.goal_info[2], min=-1.0, max=1.0)
        # Multiply the goal_info to fit in the same range of the CNN filters
        self.goal_info = self.navigation_multiplyer * self.goal_info
        # print("     [distance, cosine, sine]: ", self.goal_info[0, :, :], self.goal_info[1, :, :],
        #       self.goal_info[2, :, :])
        # print("     ", self.goal_info[0,:,:], self.goal_info[1,:,:], self.goal_info[2,:,:])
        # Append the goal information (distance and bearing) to the observation space
        # last_ob = np.append(last_ob, self.goal_info, axis=1)

        #
        if self.use_stack_memory is True and self.use_depth is False:
            # TODO: modify to take care of stack of observation + depth
            for i in range(self.img_channels -1, 0, -1):
                self.obs[:,:, i] = self.obs[:,:,i-1]
                # self.obs[:,:,3] = self.obs[:,:,2]
                # self.obs[:,:,2] = self.obs[:,:,1]
                # self.obs[:,:,1] = self.obs[:,:,0]

        # If you want to use only depth image
        if self.use_depth is True and self.use_combined_depth is False:
            self.obs[:, :, 0] = np.reshape(np.append(last_depth, self.goal_info, axis=1), newshape=(84, 85))
        else:
            self.obs[:,:,0] = np.reshape(np.append(last_ob, self.goal_info, axis=1), newshape=(84,85))

        if self.use_combined_depth is True:
            self.obs[:, :, -1] = np.reshape(np.append(last_depth, self.goal_info, axis=1), newshape=(84, 85))

        #########################
        ##        REWARD       ##
        #########################
        if self.done == False:
            self.reward = self.nav_utils.getReward(self.distance, self.robot_rel_orientation)
            if self.collision_detection == True:
                self.reward += self.nav_utils.getCollisionPenalty(self.last_collision)
                # Reset the last collision
                self.last_collision = None

        #########################
        ##         DONE        ##
        #########################
        # If the reward is greater than zero, we are in proximity of the goal. So Done needs to be set to true
        if self.reward >= 0 or self.iterator == (self.max_episode_steps - 1) or self.reward <= self.tolerance_penalty:
            self.done = True
            self.pose_acceptable = False


        # Printing info
        if self.done == True:
            # rospy.logwarn("Done: " + str(self.done))
            print("Final distance to goal: ", self.distance)
            if self.reward >= 0:
                rospy.logwarn(" -> The robot reached its target.")
            elif self.reward <= self.tolerance_penalty:
                rospy.logerr("  -> The robot crashed into a static obstacle.")
            else:
                pass

        # self.time_stop = float(time.time())
        # self.rospy_time_stop = rospy.get_rostime()
        # print("[B]Time difference between step: ", (float(self.time_stop - self.time_start)), " sec")
        # print("[B]ROSPY Time difference between step: ", self.rospy_time_stop - self.rospy_time_start, " nsec")

        return self.obs, self.reward, self.done, {}
        # return last_ob, self.reward, self.done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.

        # Reset the step iterator
        self.steps_counter += self.iterator
        self.iterator = 0
        self.episodes_counter += 1

        print("====== New episode: {} [{}] ======".format(self.episodes_counter, self.steps_counter))


        if self.synch_mode == True:
            # Unpause simulation
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
                # print("UnPausing")
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

        # TODO: wait_for_message dies after a while. The check should be implemented using normal subscribers
        # while self.collision_detection == True and msg == None:
        #     # print("Waiting")
        #     msg = rospy.wait_for_message("/collision_data", ContactState)

        if self.reset_position is True:
            rospy.wait_for_service('/gazebo/set_model_state')
            # print("Waiting for object list")
            self.objects_list = None
            while self.objects_list == None:
                self.objects_list = rospy.wait_for_message("/gazebo/model_states", ModelStates)

            # Once in a while, reset objects position
            # TODO: regenerate the world if the robot cannot find a new position for X times
            if self.episodes_counter % self.episodes_reset == 0:
                print("First, clean the environment removing the objects")
                for i in range(0, len(self.objects_list.name)):
                    if self.objects_list.name[i] == "ground_list":
                        pass
                    else:
                        # For every object, find a new random position and check if it's available
                        tmp_pose = self.nav_utils.getRandomPosition(reference_frame=self.reference_frame,
                                                                     model_name=self.objects_list.name[i],
                                                                     world_size=[-25.0, -50.0, -25.0, -30.0])
                        try:
                            self.set_position_proxy(tmp_pose)
                        except (rospy.ServiceException) as e:
                            rospy.logerr("/gazebo/set_model_state service call failed")
                print("Second, generate random position...")
                for i in range(0, len(self.objects_list.name)):
                    if self.objects_list.name[i] == "ground_list":
                        pass
                    else:
                        # For every object, find a new random position and check if it's available
                        # print("Waiting for object list")
                        self.objects_list = None
                        while self.objects_list == None:
                            self.objects_list = rospy.wait_for_message("/gazebo/model_states", ModelStates)
                        self.pose_acceptable = False
                        if self.objects_list.name[i] != self.model_name:
                            while self.pose_acceptable == False:
                                # print("[{}] generating pose".format(i))
                                tmp_pose = self.nav_utils.getRandomPosition(reference_frame=self.reference_frame,
                                                                            model_name=self.objects_list.name[i],
                                                                            world_size=self.world_xy)
                                self.pose_acceptable = self.nav_utils.checkPose(tmp_pose, self.objects_list)
                            # if self.objects_list.name[i] != self.model_name:
                            #     self.initial_pose = tmp_pose
                            try:
                                self.set_position_proxy(tmp_pose)
                            except (rospy.ServiceException) as e:
                                rospy.logerr("/gazebo/set_model_state service call failed")
                        # rospy.sleep(2)

            # Reset robot pose at every episode
            # print("Resetting robot position")
            self.pose_acceptable = False
            self.objects_list = None
            while self.objects_list == None:
                self.objects_list = rospy.wait_for_message("/gazebo/model_states", ModelStates)

            while self.pose_acceptable == False:
                # print("Selecting random pose")
                self.initial_pose = self.nav_utils.getRandomPosition(reference_frame=self.reference_frame,
                                                                     model_name=self.model_name,
                                                                     world_size=self.robot_xy)
                self.pose_acceptable = self.nav_utils.checkPose(self.initial_pose, self.objects_list)
            try:
                self.set_position_proxy(self.initial_pose)
            except (rospy.ServiceException) as e:
                rospy.logerr("/gazebo/set_model_state service call failed")

            # print("New initial position: ", self.initial_pose)
            # rospy.sleep(2.0)


        # Take an observation
        last_ob = None
        last_depth = None
        while (last_ob is None):
            try:
                if self.fake_images == True:
                    last_ob = np.zeros(shape=(84, 84, 1))  # DEBUG
                else:
                    last_ob = self.takeObservation()
            except:
                rospy.logerr("Problems acquiring the observation")

        while (last_depth is None):
            try:
                last_depth = self.takeDepth()
            except:
                rospy.logerr("Problems acquiring the observation")


        # print("Observation acquired!")
        # Reset the target position and calculate distance robot-target
        self.target_position = self.nav_utils.getRandomTargetPosition(self.initial_pose)
        self.robot_abs_pose = self.nav_utils.getRobotAbsPose()
        self.distance = self.nav_utils.getGoalDistance(self.robot_abs_pose, self.target_position)
        self.goal_info[0] = self.nav_utils.normalise(value=self.distance, min=0.0, max=self.max_distance)
        # print("Distance-Goal found!")
        self.euler_bearing = self.nav_utils.getBearingEuler(self.robot_abs_pose)
        # self.goal_info[1] = self.euler_bearing[1]  # assuming (R,Y, P)
        self.robot_target_abs_angle = self.nav_utils.getRobotTargetAbsAngle(self.robot_abs_pose, self.target_position)
        # self.robot_rel_orientation = np.math.radians(self.nav_utils.getRobotRelOrientation(self.robot_target_abs_angle, self.euler_bearing[1]))
        self.robot_rel_orientation = self.nav_utils.getRobotRelOrientationAtan2(self.robot_target_abs_angle,
                                                                                self.euler_bearing[1])
        self.goal_info[1] = self.robot_rel_orientation
        # print("Orientation-Goal found!")
        # print("[", self.iterator, "]Angle: ", np.degrees(self.goal_info[1]))
        if self.use_cosine_sine == True:
            self.goal_info[1] = math.cos(self.robot_rel_orientation)  # angles must be expressed in radiants
            self.goal_info[2] = math.sin(self.robot_rel_orientation)
            # print("     [distance, cosine, sine]: ", self.goal_info[0,:,:], self.goal_info[1,:,:], self.goal_info[2,:,:])
            # Normalise the sine and cosine
            # self.goal_info[1] = self.nav_utils.normalise(value=self.goal_info[1], min=-1.0, max=1.0)
            # self.goal_info[2] = self.nav_utils.normalise(value=self.goal_info[2], min=-1.0, max=1.0)
        # Multiply the goal_info to fit in the same range of the CNN filters
        self.goal_info = self.navigation_multiplyer * self.goal_info
        # print("     [distance, cosine, sine]: ", self.goal_info[0,:,:], self.goal_info[1,:,:], self.goal_info[2,:,:])
        # Append the goal information (distance and bearing) to the observation space
        # last_ob = np.append(last_ob, self.goal_info, axis=1)


        # If you want to use only depth image
        if self.use_depth is True and self.use_combined_depth is False:
            self.obs[:, :, 0] = np.reshape(np.append(last_depth, self.goal_info, axis=1), newshape=(84, 85))
        else:
            self.obs[:, :, 0] = np.reshape(np.append(last_ob, self.goal_info, axis=1), newshape=(84, 85))

        # TODO: modify to take care of stack of observation + depth
        if self.use_stack_memory is True and self.use_depth is False:
            for i in range(1, self.img_channels):
                self.obs[:,:,i] = self.obs[:,:,0]

        if self.use_combined_depth is True:
            self.obs[:, :, -1] = np.reshape(np.append(last_depth, self.goal_info, axis=1), newshape=(84, 85))

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

        return self.obs
        # return last_ob
