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
import rospy


# from baselines import logger
# from baselines.common import set_global_seeds, tf_util as U
# from baselines.ppo1 import mlp_policy, pposgd_simple, cnn_policy
# from baselines import bench

from stable_baselines import logger
from stable_baselines.common import set_global_seeds, tf_util as U
from stable_baselines import PPO1, PPO2, TRPO
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy, NavigationCnnPolicy, FeedForwardPolicy, NavigationMlpPolicy, NavigationCnnLstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc




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


best_mean_reward, n_steps = -np.inf, 5000

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  # TODO: The callback is not called, it may not work. MUST be fixed
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(directory), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(directory + 'best_model.pkl')
  n_steps += 1
  return True

###########################
#         MODEL           #
###########################
# GazeboThorvaldCameraEnv-v0: single camera
# GazeboThorvaldCameraEnv-v1: multicameras
# GazeboThorvaldLidarEnv-v0: lidar

env = gym.make('GazeboThorvaldCameraEnv-v0')

directory="/home/pulver/Desktop/Experiments/Avoidance/depth/singlecamera/no_big_reward/run-4/"
# directory="/tmp/ppo/"
ckp_path = directory + "run-4.pkl"

try:
    os.makedirs(directory)
    print("Directory " , directory ,  " created ")
except FileExistsError:
    pass
env = Monitor(env, directory, allow_early_resets=True)
env = DummyVecEnv([lambda : env])  # The algorithm require a vectorized environment to run

num_timesteps = 100000
test_episodes = 100
model = PPO2(NavigationCnnPolicy, env=env, n_steps=800, verbose=1, tensorboard_log=directory, full_tensorboard_log=True)
# model = PPO2(NavigationCnnLstmPolicy, env=env, n_steps=20, nminibatches=1,  verbose=1, tensorboard_log=directory, full_tensorboard_log=True)
seed = np.random.randint(low=0, high=5)
seed = 0
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
# os.environ[ 'OPENAI_LOGDIR' ] = directory
# os.environ[ 'OPENAI_LOG_FORMAT' ] = 'stdout,tensorboard'
# #
# from stable_baselines import logger
# print( 'Configuring stable-baselines logger')
# logger.configure()
# env = bench.Monitor(env, logger.get_dir())
timer_start = time.time()
###########################
#         TRAIN           #
###########################
if test is False:
    print("====== TRAIN ======")
    print("Saving file in: ", directory)
    print("Seed used: ", seed)
    # model_1.learn(total_timesteps=3e5, tb_log_name="999")
    model.learn(total_timesteps=num_timesteps, seed=seed, callback=callback)
    model.save(save_path=ckp_path)
    print("Saving")
    # del model
    # model = PPO1.load(ckp_path + ".pkl", env, tensorboard_log=directory)
    # print("Loading")
    # model.learn(total_timesteps=num_timesteps)
    # model.save(save_path=ckp_path + "2")

else:
###########################
#         TEST           #
###########################
    print("====== TEST ======")
    # Load the trained agent
    ckp_path_list = ["run-1/run-1.pkl",
                     "run-2/run-2.pkl",
                     "run-3/run-3.pkl",
                     "run-4/run-4.pkl"]
    ckp_results = [None] * len(ckp_path_list)
    ckp_counter = 0
    obs = env.reset()
    for ckp in ckp_path_list:
        ckp_path = directory + ckp
        print("------------------")
        print("Loading ckp from: ", ckp_path)
        print("------------------")
        model = PPO2.load(ckp_path, env, tensorboard_log=directory)
        # Enjoy trained agent
        success = 0
        max_episode_steps = 200
        for episodes in range(test_episodes):
            for step in range(max_episode_steps):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                if done[0] == True:
                    outcome = 0
                    if rewards > 0:
                        success += 1
                        outcome = 1
                    # print("Writing to log...")
                    with open(directory + "New_results_test_100.csv", "a") as myfile:
                        string_to_add = ckp + "," + str(episodes) + "," + str(step) + "," + str(outcome) + "\n"
                        myfile.write(string_to_add)
                    # NB: VecEnv reset the environment automatically when done is True
                    break
                # env.render()  # Not required when using Gazebo
        ckp_results[ckp_counter] = success / test_episodes
        ckp_counter += 1
    # Print ckp and respective success rate
    print("------------------")
    print("Success rate")
    for i in range(len(ckp_path_list)):
        print("{}:{}".format(ckp_path_list[i], ckp_results[i]))
        with open(directory + "New_success_rate_100.txt", "a") as myfile:
            string_to_add = str(ckp_path_list[i]) + ":" + str(ckp_results[i]) + "\n"
            myfile.write(string_to_add)
    print("------------------")


timer_stop = time.time()
sec = timer_stop - timer_start
print("=======================")
print("Time simulation: {}s, {}m, {}h".format(sec, sec/60.0, sec/3600))
print("=======================")