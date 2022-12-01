from gym.envs.registration import register

register(
    id='SongBirdRL',
    entry_point='SongBirdRL.envs:SongBirdEnv',
)