# SongbirdActorCritic

To create the environment, download conda build the environment from the environment.yml file using

    conda env create --name songbird_rl --file=environments.yml python=3.7.10
   
The environment runs on 3.7.10 and may or may not work with other versions of Python.

To run A2C, 

To run PPO, 

    python3 SongBird_env_PPO.py | grep ep_rew_mean > ppo_outputs/ppo_7len_20wrong.txt 
    
The name of the file choice is arbitrary. Change the action space variables (song length and wrong notes) in the corresponding env function inside. Change the plot to match.
To recreate the plots, open the plots.ipynb file and insert your filename into the plot code.

