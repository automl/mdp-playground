import ray
from ray import tune
from ray.rllib.utils.seed import seed as rllib_seed
import rl_toy
from rl_toy.envs import RLToyEnv
from ray.tune.registry import register_env
register_env("RLToy-v0", lambda config: RLToyEnv(config))
# import gym
# gym.make("RLToy-v0")
print(type(RLToyEnv))

# rllib_seed(0, 0, 0)
ray.init()
tune.run(
    "DQN",
    stop={
        "timesteps_total": 20000,
          },
    config={
      "adam_epsilon": 0.00015,
      "beta_annealing_fraction": 1.0,
      "buffer_size": 1000000,
      "double_q": False,
      "dueling": False,
      "env": "RLToy-v0",
      "env_config": {
        'state_space_type': 'discrete',
        'action_space_type': 'discrete',
        'state_space_size': 16,
        'action_space_size': 16,
        'generate_random_mdp': True,
        'delay': 6,
        'sequence_length': 1,
        'reward_density': 0.25,
        'terminal_state_density': 0.1
        },
      "exploration_final_eps": 0.01,
      "exploration_fraction": 0.1,
      "final_prioritized_replay_beta": 1.0,
      "hiddens": [
        256
      ],
      "learning_starts": 2000,
      "lr": 6.25e-05,
      "n_step": 1,
      "noisy": False,
      "num_atoms": 1,
      "prioritized_replay": False,
      "prioritized_replay_alpha": 0.5,
      "sample_batch_size": 4,
      "schedule_max_timesteps": 20000,
      "target_network_update_freq": 80,
      "timesteps_per_iteration": 100,
      "train_batch_size": 32
    },
)
