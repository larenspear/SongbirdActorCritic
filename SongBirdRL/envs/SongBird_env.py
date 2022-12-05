import gym
import numpy as np
from gym import spaces


class SongBirdEnv(gym.Env):
    """An OpenAI gym environment for song learning"""
    metadata = {'render.modes': ['human']}

    def __init__(self,song_length=4,num_error_notes=5,song=[-1,4,2,1,3]):
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

        obs = np.array([self.current_step])
        reward = self._get_reward(action)
        self.total_reward += reward
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
      print(f"Episode Number: {self.episode_num}")
      print(self.episode)
      print(f"Total Reward: {self.total_reward}")
      self.episode = [-1]
      self.episode_num += 1
      self.current_step = 0
      obs = np.array([self.current_step])
      self.total_reward = 0


      return obs


    def render(self, mode='human', close=False):
        # Render the environment to the screen
      pass