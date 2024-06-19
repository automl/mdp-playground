from gymnasium.envs.registration import register

register(
    id="RLToy-v0",
    entry_point="mdp_playground.envs:RLToyEnv",
)

register(
    id="RLToyFiniteHorizon-v0",
    entry_point="mdp_playground.envs:RLToyEnv",
    max_episode_steps=100,
)

__version__ = "1.0.0"
