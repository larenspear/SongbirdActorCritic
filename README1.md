# SongbirdActorCritic

To create the environment, download conda build the environment from the environment.yml file using

    conda env create --name envname --file=environments.yml
   
To run A2C, 

To run PPO, 

    python3 SongBird_env_PPO.py | grep ep_rew_mean > ppo_7len_20wrong.txt 
    
The name of the file choice is arbitrary. Change the action space variables (song length and wrong notes) in the corresponding env function inside. Change the plot to match.
To recreate the plots, open the plots.ipynb file and insert your filename into the plot code.
