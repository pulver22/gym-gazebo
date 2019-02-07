#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import numpy as np
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
from stable_baselines import PPO1, PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnPolicy, NavigationCnnPolicy, FeedForwardPolicy, NavigationMlpPolicy


# def policy_cnn(name, env):
#     return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space, feature_extraction="navigation_cnn")
# if __name__ == '__main__':







###########################
#         MODEL           #
###########################

env = gym.make('GazeboThorvaldCameraEnv-v1')  # Camera + nav_info
env = DummyVecEnv([lambda : env])  # The algorithm require a vectorized environment to run
# env = gym.make('GazeboThorvaldMlpEnv-v1')  # Only nav_info
print("----  Environment action limits: ", env.action_space.low,", ",  env.action_space.high)
seed = 0
directory="/home/pulver/Desktop/ppo_thorvald/no_cos_norm/NAVCNN/"
# directory="/home/pulver/Desktop/test_clock/"
ckp_path = directory + "no_cos_norm_relu"
num_timesteps = 300000
test_episodes = 10
model = PPO2(NavigationCnnPolicy, env=env, verbose=1, tensorboard_log=directory, max_grad_norm=0.5)
# model = PPO1(NavigationCnnPolicy, env, verbose=1, timesteps_per_actorbatch=800,  tensorboard_log=directory)
# model = PPO1(NavigationMlpPolicy, env, verbose=1, timesteps_per_actorbatch=800,  tensorboard_log=directory)
test = False

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
#         TRAIN           #
###########################
if test == False:
    timer_start = time.time()
    #model_1.learn(total_timesteps=3e5, tb_log_name="999")
    model.learn(total_timesteps=num_timesteps)
    model.save(save_path=ckp_path)
    print("Saving")
    # del model
    # model = PPO1.load(ckp_path + ".pkl", env, tensorboard_log=directory)
    # print("Loading")
    # model.learn(total_timesteps=num_timesteps)
    # model.save(save_path=ckp_path + "2")
    timer_stop = time.time()
    print("Time simulation: " + str(timer_stop - timer_start) + " seconds")
else:
###########################
#         TEST           #
###########################
    # Load the trained agent
    model = PPO1.load(ckp_path + ".pkl", env, tensorboard_log=directory)
    # Enjoy trained agent
    obs = env.reset()
    for episodes in range(test_episodes):
        env.reset()
        for i in range(200):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if rewards > 0:
                break
            #env.render()  # Not required when using Gazebo
