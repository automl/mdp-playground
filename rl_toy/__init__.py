from gym.envs.registration import register

register(
    id='RLToy-v0',
    entry_point='rl_toy.envs:RLToyEnv',
#    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
#    max_episode_steps = 200,
)
