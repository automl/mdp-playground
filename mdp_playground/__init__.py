from gym.envs.registration import register

register(
    id='RLToy-v0',
    entry_point='mdp_playground.envs:RLToyEnv',
)
