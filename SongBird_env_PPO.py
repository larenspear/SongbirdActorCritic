import gym
import numpy as np
from gym import spaces
from itertools import count
import os
from stable_baselines.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DDPG, SAC
from stable_baselines3.common.logger import configure, CSVOutputFormat
from stable_baselines3.common import logger
##
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold

import matplotlib.pyplot as plt
##

tmp_path = "./tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "log"])

class SongBirdEnv(gym.Env):
    """An OpenAI gym environment for song learning"""
    metadata = {'render.modes': ['human']}

    def __init__(self,song_length=7,num_error_notes=20,song=[-1,4,2,1,3,2,1,3]):
        super(SongBirdEnv, self).__init__()
        
        self.num_error_notes = num_error_notes
        self.num_correct_notes = 4 # A,B,C,D
        self.init_notes = 1 # i 
        self.song_length = song_length # Can be between 3-7

        self.action_space =  spaces.Discrete(self.num_error_notes+self.num_correct_notes+self.init_notes)
 
        #self.observation_space = spaces.Discrete(self.song_length+1)
        self.observation_space = spaces.Box(low=0, high=self.song_length, shape=(1,1), dtype=np.int32)
        self.episode = [-1]
        self.current_step = 0
        self.episode_num = 0
        self.song = song #TODO 
        self.prediction = [-1 for _ in range (len(self.song))]
        self.previous_attempt = [False for _ in range (len(self.song))]
        self.total_reward = 0

    def _take_action(self, action):
    
      self.episode.append(action)

    def _get_reward(self,action): ##Models Dopamine Secretion in Song Birds
       reward = 0
       if (self.song[self.current_step] == action):
          correct_action = True
       else:
          correct_action = False
       if (self.prediction[self.current_step] == action):
          correct_prediction = True
       else:
          correct_prediction = False 
       
       if (correct_action and correct_prediction):
           if(self.previous_attempt[self.current_step]):
              reward = 5
           else:
              reward = 10 
              self.previous_attempt[self.current_step] = True

       elif (correct_action and (not correct_prediction)):
           reward = 10
           self.prediction[self.current_step]= action
           self.previous_attempt[self.current_step] = True
       elif ((not correct_action) and correct_prediction):
           reward = 0
           self.previous_attempt[self.current_step] = False
       return reward

    def step(self, action):
        # Execute one time step within the environment
        #if (self.current_step > self.observation_space.shape[0]):
        #    raise RuntimeError("Episode is done")
        self._take_action(action)

        self.current_step += 1
        if (self.current_step == self.observation_space.high):
          done = True
        else:
          done = False 

        obs = np.array([self.current_step]).reshape((1,1))
        obs = obs.astype(np.int32)
        reward = self._get_reward(action)
        self.total_reward += reward
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
      #print(self.episode_num, ', ', self.total_reward,sep='')
      #print(f"Episode Number: {self.episode_num}")
      #print(self.episode)
      #print(f"Total Reward: {self.total_reward}")
      self.episode = [-1]
      self.episode_num += 1
      self.current_step = 0
      obs = np.zeros((1,1),dtype=np.int32)
      self.total_reward = 0

      return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
      pass

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

env = SongBirdEnv()
check_env(env, warn=True)
#log_dir = "."
#env = Monitor(env, log_dir)
model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=10, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, tensorboard_log=None, policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True)

#logger.configure(folder=".", format_strings=["csv"]) 

eval_env = SongBirdEnv()
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=25, verbose=1)
eval_callback = EvalCallback(eval_env, log_path="./logs/",callback_on_new_best=callback_on_best, eval_freq=10, verbose=1)

model.learn(total_timesteps=50000)#callback=eval_callback)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
