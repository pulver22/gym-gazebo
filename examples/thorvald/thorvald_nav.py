#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import numpy as np
import gym_gazebo  # Needed for resolving environment name
import time
import os


from stable_baselines import logger
from stable_baselines import PPO1, PPO2, TRPO
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.policies import CnnPolicy, NavigationCnnPolicy, FeedForwardPolicy, NavigationMlpPolicy, NavigationCnnLstmPolicy, LstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from gym_gazebo.utils.custom_networks import NavigationCnnPolicy


###########################
#          VARs           #
###########################
best_mean_reward, n_steps = -np.inf, 4999
seed = 0
# directory='/media/pulver/PulverHDD/Experiments/Avoidance/greyscale/singlecamera/fullobservability/singlebox/ref1/run-' + str(seed) +'/'
# directory = '/media/pulver/PulverHDD/Experiments/Avoidance/rgb/singlecamera/fullobservability/combined/ref1/big_box/box_reward_0/alternate_20ep/run-' + str(seed) + '/'
directory = '/media/pulver/PulverHDD/Experiments/Avoidance/rgb/singlecamera/fullobservability/combined/ref1/cilinder/cilinder_reward_0/alternate_10ep/5kg/fixed/run-' + str(seed) + '/'
#seed = np.random.randint(low=0, high=8)
# directory='/tmp/ppo/test/'
ckp_path = directory + 'run-' + str(seed) + '.pkl'
try:
    os.makedirs(directory)
    print('Directory ' , directory ,  ' created ')
except FileExistsError:
    pass


#TODO: I believe this is not called
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 5000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(directory), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print('Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}'.format(best_mean_reward, mean_reward))
          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print('Saving new best model')
              _locals['self'].save(directory + 'best_model.pkl')
  n_steps += 1
  return True

###########################
#           ENV           #
###########################
# GazeboThorvaldCameraEnv-v0: single camera
# GazeboThorvaldCameraEnv-v1: multicameras
# GazeboThorvaldLidarEnv-v0: lidar
env = gym.make('GazeboThorvaldCameraEnv-v0')


env = Monitor(env, directory, allow_early_resets=True)
env = DummyVecEnv([lambda : env])  # The algorithm require a vectorized environment to run


###########################
#          MODEL          #
###########################
num_timesteps = 200000
test_episodes = 30
model = PPO2(NavigationCnnPolicy, env=env, n_steps=800, verbose=1, tensorboard_log=directory, full_tensorboard_log=True)


test = False
continual_learning = False
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
    print('====== TRAIN ======')
    if continual_learning is False:
        print('Saving file in: ', directory)
        print('Seed used: ', seed)
        model.learn(total_timesteps=num_timesteps, seed=seed, callback=callback)
        model.save(save_path=ckp_path)
        print('Saving')
    else:
        print('------------------')
        print('Loading ckp from: ', ckp_path)
        print('------------------')
        model = PPO2.load(ckp_path, env, tensorboard_log=directory)
        model.learn(total_timesteps=num_timesteps, seed=seed, callback=callback)
        ckp_path = directory + 'run-' + str(seed) + '4.pkl'
        model.save(save_path=ckp_path)
        print('Saving')
else:
###########################
#         TEST           #
###########################
    print('====== TEST ======')
    # Load the trained agent
    ckp_path_list = ["run-02.pkl"]#,
                      # "run-0.pkl2",
                      # "run-0.pkl23"]#,
                      # "run-3/run-3.pkl"]
                     # "run-4/run-4.pkl",
                     # "run-5/run-5.pkl",
                     # "run-6/run-6.pkl",
                     # "run-7/run-7.pkl",
                     # "run-8/run-8.pkl"]
    ckp_results = [None] * len(ckp_path_list)
    ckp_counter = 0
    obs = env.reset()
    for ckp in ckp_path_list:
        ckp_path = directory + ckp
        print('------------------')
        print('Loading ckp from: ', ckp_path)
        print('------------------')
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
                    print('Writing to log...')
                    with open(directory + 'Results_test.csv', 'a') as myfile:
                        string_to_add = ckp + ',' + str(episodes) + ',' + str(step) + ',' + str(outcome) + '\n'
                        myfile.write(string_to_add)
                    # NB: VecEnv reset the environment automatically when done is True
                    break
                # env.render()  # Not required when using Gazebo
        ckp_results[ckp_counter] = success / test_episodes
        ckp_counter += 1
    # Print ckp and respective success rate
    print('------------------')
    print('Success rate')
    for i in range(len(ckp_path_list)):
        print('{}:{}'.format(ckp_path_list[i], ckp_results[i]))
        with open(directory + 'Success_rate.txt', 'a') as myfile:
            string_to_add = str(ckp_path_list[i]) + ':' + str(ckp_results[i]) + '\n'
            myfile.write(string_to_add)
    print('------------------')


timer_stop = time.time()
sec = timer_stop - timer_start
print('=======================')
print('Time simulation: {}s, {}m, {}h'.format(sec, sec/60.0, sec/3600))
print('=======================')
