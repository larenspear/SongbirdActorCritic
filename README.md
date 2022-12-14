# SongbirdActorCritic

To create the environment, download conda build the environment from the environment.yml file using

    conda env create --name songbird_rl --file=environments.yml python=3.7.10
   
The environment runs on 3.7.10 and may or may not work with other versions of Python.

To install the SongbirdRL environment, do the following:
 git clone https://github.com/larenspear/SongbirdActorCritic.git
 cd to SongBirdRL directory
 pip install -e .
 
 To create a SonBirdRL environment, you have to use the following code:
 
   env = gym.make('SongBirdRL-v0',song_length=4,num_error_notes=5,song=[-1,4,2,1,3],max_reward_per_note=10,baseline_reward_per_note=5,gamma_across_episode=0.99)
   
   You can use any song length and song, but song length should match length of the song list + 1 (For init note in the beginning)

To run A2C,

   python actor-critic.py --gae 1
   
To run A2C with successive discounting,

   python actor-critic.py --gae 0.9
   
To run PPO, 

    python3 SongBird_env_PPO.py | grep ep_rew_mean > ppo_outputs/ppo_7len_20wrong.txt 
    
The name of the file choice is arbitrary. Change the action space variables (song length and wrong notes) in the corresponding env function inside. Change the plot to match.
To recreate the plots, open the plots.ipynb file and insert your filename into the plot code.

