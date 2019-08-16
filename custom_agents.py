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
        "tolerance": 0.01,
        "discount_factor": 0.5,
        # "lr": 0.5
    })

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])
        self.V = np.zeros(self.env.observation_space.n)
        self.policy = np.zeros(self.env.observation_space.n, dtype=int)
        self.policy[:] = -1 #IMP # To avoid initing it to a value within action_space range

    @override(Trainer)
    def _train(self):
        max_diff = np.inf # Maybe keep a state variable so that we don't need to update every train iteration??
        state_space_size = self.env.observation_space.n
        gamma = self.config["discount_factor"]
        total_iterations = 0
        while max_diff > self.config["tolerance"]:
            total_iterations += 1
            for s in range(state_space_size):
                # print("self.V[:]", s, max_diff, self.V, [self.env.R(s, a) for a in range(self.env.action_space.n)], self.policy[s])
                self.V_old = self.V.copy() # Is this asynchronous? V_old should be held constant for all states in the for loop?
                # print([self.env.R(s, a) for a in range(self.env.action_space.n)], [gamma * self.V[self.env.P(s, a)] for a in range(self.env.action_space.n)], [self.env.R(s, a) + gamma * self.V[self.env.P(s, a)] for a in range(self.env.action_space.n)])
                self.policy[s] = np.argmax([self.env.R(s, a) + gamma * self.V[self.env.P(s, a)] for a in range(self.env.action_space.n)])
                self.V[s] = np.max([self.env.R(s, a) + gamma * self.V[self.env.P(s, a)] for a in range(self.env.action_space.n)]) # We want R to be a callable function, so I guess we have to keep a for loop here??
                # print("self.V, self.V_old, self.policy[s]", self.V, self.V_old, self.policy[s], self.env.P(s, self.policy[s]))

                max_diff = np.max(np.absolute(self.V_old - self.V))
        # import time
        # time.sleep(2)
        # for s in range(state_space_size):
        #     print("FINAL self.V[:]", s, max_diff, self.V[:], [self.env.R(s, a) for a in range(self.env.action_space.n)])

        rewards = []
        steps = 0
        for _ in range(10):
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
# tune.run(
#     RandomAgent,
#     stop={
#         "timesteps_total": 20000,
#           },
#     config={
#       "rollouts_per_iteration": 10,
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
#     },
# )

tune.run(
    VIAgent,
    stop={
        "timesteps_total": 20000,
          },
    config={
      # "rollouts_per_iteration": 10,
      "env": "RLToy-v0",
      "env_config": {
        'state_space_type': 'discrete',
        'action_space_type': 'discrete',
        'state_space_size': 10,
        'action_space_size': 10,
        'generate_random_mdp': True,
        'delay': 0,
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
