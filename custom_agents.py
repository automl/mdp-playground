from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.utils.annotations import override


# yapf: disable
# __sphinx_doc_begin__
class RandomAgent(Trainer):
    """Policy that takes random actions and never learns."""

    _name = "RandomAgent"
    _default_config = with_common_config({
        "rollouts_per_iteration": 10,
    })

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])

    @override(Trainer)
    def _train(self):
        rewards = []
        steps = 0
        for _ in range(self.config["rollouts_per_iteration"]):
            obs = self.env.reset()
            done = False
            reward = 0.0
            while not done:
                action = self.env.action_space.sample()
                obs, r, done, info = self.env.step(action)

                reward += r
                steps += 1
            rewards.append(reward)
        return {
            "episode_reward_mean": np.mean(rewards),
            "timesteps_this_iter": steps,
        }

class VIAgent(Trainer):
    """Value Iteration."""

    _name = "VIAgent"
    _default_config = with_common_config({
        "rollouts_per_iteration": 10,
    })

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])

    @override(Trainer)
    def _train(self):
        rewards = []
        steps = 0
        for _ in range(self.config["rollouts_per_iteration"]):
            obs = self.env.reset()
            done = False
            reward = 0.0
            while not done:
                action = self.env.action_space.sample()
                obs, r, done, info = self.env.step(action)

                reward += r
                steps += 1
            rewards.append(reward)
        return {
            "episode_reward_mean": np.mean(rewards),
            "timesteps_this_iter": steps,
        }

# __sphinx_doc_end__
# don't enable yapf after, it's buggy here


# def _import_random_agent():
#     # from ray.rllib.contrib.random_agent.random_agent import RandomAgent
#     return RandomAgent
#
# from ray.rllib.agents.registry import CONTRIBUTED_ALGORITHMS
#
# CONTRIBUTED_ALGORITHMS["contrib/RandomAgent"] = _import_random_agent,



import ray
from ray import tune
from ray.rllib.utils.seed import seed as rllib_seed
import rl_toy
from rl_toy.envs import RLToyEnv
from ray.tune.registry import register_env
register_env("RLToy-v0", lambda config: RLToyEnv(config))

# rllib_seed(0, 0, 0)
ray.init()
tune.run(
    RandomAgent,
    stop={
        "timesteps_total": 20000,
          },
    config={
      "rollouts_per_iteration": 10,
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
        'terminal_state_density': 0.25
        },
    },
)

# tune.run(
#     "DQN",
#     stop={
#         "timesteps_total": 20000,
#           },
#     config={
#       "adam_epsilon": 0.00015,
#       "beta_annealing_fraction": 1.0,
#       "buffer_size": 1000000,
#       "double_q": False,
#       "dueling": False,
#       "env": "RLToy-v0",
#       "env_config": {
#         'state_space_type': 'discrete',
#         'action_space_type': 'discrete',
#         'state_space_size': 16,
#         'action_space_size': 16,
#         'generate_random_mdp': True,
#         'delay': 6,
#         'sequence_length': 1,
#         'reward_density': 0.25,
#         'terminal_state_density': 0.25
#         },
#       "exploration_final_eps": 0.01,
#       "exploration_fraction": 0.1,
#       "final_prioritized_replay_beta": 1.0,
#       "hiddens": [
#         256
#       ],
#       "learning_starts": 2000,
#       "lr": 6.25e-05,
#       "n_step": 1,
#       "noisy": False,
#       "num_atoms": 1,
#       "prioritized_replay": False,
#       "prioritized_replay_alpha": 0.5,
#       "sample_batch_size": 4,
#       "schedule_max_timesteps": 20000,
#       "target_network_update_freq": 80,
#       "timesteps_per_iteration": 100,
#       "train_batch_size": 32
#     },
# )



if __name__ == "__main__":
    trainer = RandomAgent(
        env="CartPole-v0", config={"rollouts_per_iteration": 10})
    result = trainer.train()
    assert result["episode_reward_mean"] > 10, result
    print("Test: OK")
