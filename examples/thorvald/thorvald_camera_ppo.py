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
from stable_baselines.common.policies import CnnPolicy


def policy_cnn(name, ob_space, ac_space):
    return cnn_policy.CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space, kind='large')
# if __name__ == '__main__':


env = gym.make('GazeboThorvaldCameraEnv-v0')  # Camera + CNN
# env = bench.Monitor(env, logger.get_dir())

seed = 0
#env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
model_1 = PPO1(CnnPolicy, env, verbose=1, timesteps_per_actorbatch=999,  tensorboard_log="/home/pulver/Desktop/ppo_turtlebot/model_1")
model_2 = PPO1(CnnPolicy, env, verbose=1, timesteps_per_actorbatch=1998,  tensorboard_log="/home/pulver/Desktop/ppo_turtlebot/model_2")
model_3 = PPO1(CnnPolicy, env, verbose=1, timesteps_per_actorbatch=3997,  tensorboard_log="/home/pulver/Desktop/ppo_turtlebot/model_3")
model_3 = PPO1(CnnPolicy, env, verbose=1, timesteps_per_actorbatch=9999,  tensorboard_log="/home/pulver/Desktop/ppo_turtlebot/model_4")
timer_start = time.time()
model_1.learn(total_timesteps=1e6)
model_2.learn(total_timesteps=1e6)
model_3.learn(total_timesteps=1e6)
model_4.learn(total_timesteps=1e6)
# ppo_agent = pposgd_simple.learn(env, policy_cnn,
#                     max_timesteps=1e6,
#                     timesteps_per_actorbatch=2048,
#                     clip_param=0.2, entcoeff=0.0,
#                     optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
#                     optim_batchsize=64, lam=0.95, schedule='linear')
timer_stop = time.time()
print("Time episode: " + str(timer_stop - timer_start) + " seconds")

