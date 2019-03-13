#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import numpy as np
import tensorflow as tf
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
from stable_baselines import PPO1, PPO2, TRPO
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy, NavigationCnnPolicy, FeedForwardPolicy, NavigationMlpPolicy, NavigationCnnLstmPolicy

from stable_baselines.a2c.utils import conv, linear, conv_to_fc
# def policy_cnn(name, env):
#     return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space, feature_extraction="navigation_cnn")
# if __name__ == '__main__':




def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")


###########################
#         MODEL           #
###########################

env = gym.make('GazeboThorvaldCameraEnv-v1')  # Camera + nav_info
# env = gym.make('GazeboThorvaldMlpEnv-v1')  # Only nav_info
env = DummyVecEnv([lambda : env])  # The algorithm require a vectorized environment to run

# TODO: LSTM requirement?
n_cpu = 2
# env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
# env = VecFrameStack(env, 4)  # The algorithm require a vectorized environment to run

print("----  Environment action limits: ", env.action_space.low,", ",  env.action_space.high)
seed = 0
# directory="/home/pulver/Desktop/ppo_thorvald/test_collision"
# directory="/home/pulver/Desktop/test_clock/old/4/1/pre-normalised/multiplyer/"
# directory="/home/pulver/Desktop/test_clock/old/4/1/pre-normalised/multiplyer/LSTM/400/"
directory="/home/pulver/Desktop/tmp/avoidance/1/"
ckp_path = directory + "4norm"

num_timesteps = 100000
test_episodes = 10
# model = TRPO(policy=NavigationCnnPolicy, env=env, timesteps_per_batch=800, verbose=1, tensorboard_log=directory)
# model = TRPO(policy=NavigationMlpPolicy, env=env, timesteps_per_batch=800, verbose=1, tensorboard_log=directory)
# model = PPO1(NavigationCnnPolicy, env, verbose=1, timesteps_per_actorbatch=800,  tensorboard_log=directory)
model = PPO2(NavigationCnnPolicy, env=env, n_steps=800, verbose=1, tensorboard_log=directory, full_tensorboard_log=True)
# model = PPO2(NavigationCnnLstmPolicy, env=env, n_steps=20, nminibatches=1,  verbose=1, tensorboard_log=directory, full_tensorboard_log=True)
test = False
# ckp_path = "/home/pulver/Desktop/ppo_thorvald/no_cos_norm/NAVCNN/300/PPO2_3/no_cos_norm_relu.pkl"
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
if test is False:
    timer_start = time.time()
    print("Saving file in: ", directory)
    # model_1.learn(total_timesteps=3e5, tb_log_name="999")
    model.learn(total_timesteps=num_timesteps, seed=seed)
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
    model = PPO2.load(ckp_path, env, tensorboard_log=directory)
    # Enjoy trained agent
    obs = env.reset()
    success = 0
    for episodes in range(test_episodes):
        env.reset()
        for i in range(200):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if rewards > 0:
                success += 1
                break
            # env.render()  # Not required when using Gazebo
    print("Success rate: ", 100*(success / test_episodes), "%")
