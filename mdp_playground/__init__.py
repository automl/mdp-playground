from gym.envs.registration import register

import mdp_playground.scripts.run_experiments

register(
    id='RLToy-v0',
    entry_point='mdp_playground.envs:RLToyEnv',
)

register(
    id='RLToyFiniteHorizon-v0',
    entry_point='mdp_playground.envs:RLToyEnv',
    max_episode_steps=100,
)

__version__ = '0.0.6'
