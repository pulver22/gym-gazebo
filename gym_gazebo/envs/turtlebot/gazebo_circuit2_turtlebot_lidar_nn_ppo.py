import gym
import rospy
import roslib
import roslaunch
import time
import numpy as np, quaternion

from gym import utils, spaces
from gym.utils import seeding

from gym_gazebo.envs import gazebo_env

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
#from tf.transformations import quaternion_from_euler




class GazeboCircuit2TurtlebotLidarNnPPOEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self._sub = rospy.Subscriber('/scan', LaserScan, self.observation_callback)
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

        # Observation space
        self.obs_dim = 100  # Number of lidar readings
        self.observation_high = np.inf*np.ones(self.obs_dim)
        self.observation_low = - self.action_high
        self.observation_space = spaces.Box(self.observation_low, self.observation_high, dtype=np.float32)

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
        self._observation_msg =  message

    def take_observation(self):
        """
        Take observation from the environment and return itself.
        """
        # min_range = 0.2
        # done = False
        obs_message = self._observation_msg
        if obs_message is None:
            return None

        # for i, item in enumerate(obs_message.ranges):
        #     if (min_range > obs_message.ranges[i] > 0):
        #         done = True
        return np.asarray(obs_message.ranges)#,done

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

        for i, item in enumerate(self.ob):
            if (min_range > self.ob[i] > 0):
                done = True

        while(self.ob is None):
            self.ob = self.take_observation()

        if not self.done:
            # Straight reward = 5, Max angle reward = 0.5
            #self.reward = round(15*(self.velocity_high[1] - abs(action[1]) + 0.0335), 2)
            self.reward = 1
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            self.reward = - 20

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
        self.ob = self.take_observation()
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