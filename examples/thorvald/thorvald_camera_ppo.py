#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json


# from baselines import logger
# from baselines.common import set_global_seeds, tf_util as U
# from baselines.ppo1 import mlp_policy, pposgd_simple, cnn_policy
# from baselines import bench

from stable_baselines import logger
from stable_baselines.common import set_global_seeds, tf_util as U
from stable_baselines import PPO1
from stable_baselines.common.policies import CnnPolicy, NavigationCnnPolicy, FeedForwardPolicy, NavigationMlpPolicy


# def policy_cnn(name, env):
#     return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space, feature_extraction="navigation_cnn")
# if __name__ == '__main__':





env = gym.make('GazeboThorvaldMlpEnv-v1')  # Only nav_info
# env = gym.make('GazeboThorvaldCameraEnv-v1')  # Camera + nav_info

###########################
#         LOGGER          #
###########################
# basedir = "/home/pulver/Desktop/ppo_thorvald/positive_reward/"
#
# try:
#     os.makedirs(basedir)
#     print("Directory " , basedir ,  " created ")
# except FileExistsError:
#     pass
#
# os.environ[ 'OPENAI_LOGDIR' ] = basedir
# os.environ[ 'OPENAI_LOG_FORMAT' ] = 'stdout,tensorboard'
#
# from stable_baselines import logger
# print( 'Configuring stable-baselines logger')
# logger.configure()
# env = bench.Monitor(env, logger.get_dir())


###########################
#         MODEL           #
###########################

seed = 0
directory="/home/pulver/Desktop/ppo_thorvald/mlp_small_reward/test_arch/"
# directory="/home/pulver/Desktop/ppo_thorvald/test_angle/"
# directory = "/home/pulver/Desktop/ppo_thorvald/test_collision"
#env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
# model_1 = PPO1(CnnPolicy, env, verbose=1, timesteps_per_actorbatch=999,  tensorboard_log="/home/pulver/Desktop/ppo_thorvald/")
# model_2 = PPO1(CnnPolicy, env, verbose=1, timesteps_per_actorbatch=500,  tensorboard_log=directory)
# model_2 = PPO1(NavigationCnnPolicy, env, verbose=1, timesteps_per_actorbatch=1000,  tensorboard_log=directory)
model_2 = PPO1(NavigationMlpPolicy, env, verbose=1, timesteps_per_actorbatch=800,  tensorboard_log=directory)


###########################
#         TRAIN           #
###########################

timer_start = time.time()
#model_1.learn(total_timesteps=3e5, tb_log_name="999")
model_2.learn(total_timesteps=5e5)
model_2.save(save_path=directory + "test_ppo")
del model_2
print("Saving")
ckp_path = directory + "test_ppo.pkl"
model_2 = PPO1.load(ckp_path, env, tensorboard_log=directory)
print("Loading")
model_2.learn(total_timesteps=5e5)
model_2.save(save_path=directory + "test_ppo_2")
timer_stop = time.time()
print("Time simulation: " + str(timer_stop - timer_start) + " seconds")
