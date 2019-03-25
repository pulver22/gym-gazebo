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


best_mean_reward, n_steps = -np.inf, 5000

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
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

env = gym.make('GazeboThorvaldCameraEnv-v1')  # Camera + nav_info
# env = gym.make('GazeboThorvaldMlpEnv-v1')  # Only nav_info

directory="/home/pulver/Desktop/avoidance/depth/curriculum/run-3"
#directory="/home/pulver/Desktop/avoidance/1/depth/run-4"
# directory="/home/pulver/Documents/Experiments/Avoidance/depth/"
ckp_path = directory + "run-3.pkl"

try:
    os.makedirs(directory)
    print("Directory " , directory ,  " created ")
except FileExistsError:
    pass
env = Monitor(env, directory, allow_early_resets=True)
env = DummyVecEnv([lambda : env])  # The algorithm require a vectorized environment to run

num_timesteps = 200000
test_episodes = 10
# model = TRPO(policy=NavigationCnnPolicy, env=env, timesteps_per_batch=800, verbose=1, tensorboard_log=directory)
# model = TRPO(policy=NavigationMlpPolicy, env=env, timesteps_per_batch=800, verbose=1, tensorboard_log=directory)
# model = PPO1(NavigationCnnPolicy, env, verbose=1, timesteps_per_actorbatch=800,  tensorboard_log=directory)
model = PPO2(NavigationCnnPolicy, env=env, n_steps=800, verbose=1, tensorboard_log=directory, full_tensorboard_log=True)
# model = PPO2(NavigationCnnLstmPolicy, env=env, n_steps=20, nminibatches=1,  verbose=1, tensorboard_log=directory, full_tensorboard_log=True)
seed = np.random.randint(low=0, high=5)
seed = 0
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
    print("Loading ckp from: ", ckp_path)
    model = PPO2.load(ckp_path, env, tensorboard_log=directory)
    # Enjoy trained agent
    success = 0
    max_episode_steps = 200
    obs = env.reset()
    for episodes in range(test_episodes):
        print("====== Episode: {}/{} ======".format(episodes+1, test_episodes))
        for i in range(max_episode_steps):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            rospy.sleep(0.5)
            # print("D: {}, R: {}".format(done, rewards))
            if done == True:
                if rewards > 0:
                    success += 1
                break
            # env.render()  # Not required when using Gazebo
    print("Success rate: ", 100*(success / test_episodes), "%")
timer_stop = time.time()
sec = timer_stop - timer_start
print("=======================")
print("Time simulation: {}s, {}m, {}h".format(sec, sec/60.0, sec/3600))
print("=======================")