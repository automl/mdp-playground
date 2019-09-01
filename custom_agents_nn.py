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
    """Value Iteration.
    #TODO Make it Generalized PI.
    """

    _name = "VIAgent"
    _default_config = with_common_config({
        "tolerance": 0.01,
        "discount_factor": 0.5,
        "rollouts_per_iteration": 10,
        "episode_length": 200,
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
#         for s in range(state_space_size):
#             print("FINAL self.V[:]", s, max_diff, self.V[:], [self.env.R(s, a) for a in range(self.env.action_space.n)])

        print("Total iterations:", total_iterations)
        rewards = []
        steps = 0
        for _ in range(self.config["rollouts_per_iteration"]):
            obs = self.env.reset()
            done = False
            reward = 0.0
            for _ in range(self.config["episode_length"]):
                action = self.policy[obs]
                obs, r, done, info = self.env.step(action)

                reward += r
                steps += 1
            rewards.append(reward)
        return {
            "episode_reward_mean": np.mean(rewards),
            "timesteps_this_iter": steps,
        }




import ray
from ray import tune
from ray.rllib.utils.seed import seed as rllib_seed
import rl_toy
from rl_toy.envs import RLToyEnv
from ray.tune.registry import register_env
register_env("RLToy-v0", lambda config: RLToyEnv(config))



from ray.rllib.models.preprocessors import OneHotPreprocessor
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_preprocessor("ohe", OneHotPreprocessor)



#rllib_seed(0, 0, 0) ####IMP Doesn't work due to multi-process I think; so use config["seed"]
ray.init()


# Old config space
# algorithms = ["DQN"]
# state_space_sizes = [2**i for i in range(4,6)]
# action_space_sizes = [2**i for i in range(1,6)]
# delays = [0] + [2**i for i in range(5)]
# sequence_lengths = [i for i in range(1,6)]
# reward_densities = [0.25] # np.linspace(0.0, 1.0, num=5)
# # make_reward_dense = [True, False]
# terminal_state_densities = [0.25] # np.linspace(0.1, 1.0, num=5)


#test basic case
# algorithms = ["DQN"]
# state_space_sizes = [10]
# action_space_sizes = [10]
# delays = [4]
# sequence_lengths = [2]
# reward_densities = [0.25] # np.linspace(0.0, 1.0, num=5)
# # make_reward_dense = [True, False]
# terminal_state_densities = [0.25] # np.linspace(0.1, 1.0, num=5)

state_space_sizes = [8]#, 10, 12, 14] # [2**i for i in range(1,6)]
action_space_sizes = [8]#2, 4, 8, 16] # [2**i for i in range(1,6)]
delays = [0] # + [2**i for i in range(4)]
sequence_lengths = [1]#, 2]#i for i in range(1,4)]
reward_densities = [0.25] # np.linspace(0.0, 1.0, num=5)
# make_reward_dense = [True, False]
terminal_state_densities = [0.25] # np.linspace(0.1, 1.0, num=5)
algorithms = ["DQN"]
#seeds = []
# Others, keep the rest fixed for these: learning_starts, target_network_update_freq, double_dqn, fcnet_hiddens, fcnet_activation, use_lstm, lstm_seq_len, sample_batch_size/train_batch_size
# More others: adam_epsilon, exploration_final_eps/exploration_fraction, buffer_size
num_layerss = [1, 2, 3, 4]
layer_widths = [8, 32, 128]
fcnet_activations = ["tanh", "relu", "sigmoid"]
learning_startss = [500, 1000, 2000, 4000, 8000]
target_network_update_freqs = [8, 80, 800]
double_dqn = [False, True]
# lstm with sequence lengths

print('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density,'
               'terminal_state_density ')
print(algorithms, state_space_sizes, action_space_sizes, delays, sequence_lengths, reward_densities, terminal_state_densities)



# stats = {}
# aaaa = 3

#TODO Write addnl. line at beginning of file for column names
# fout = open('rl_stats_temp.csv', 'a') #hardcoded
# fout.write('# basename, n_points, n_features, n_trees ')

import time
start = time.time()
print(algorithms, state_space_sizes, action_space_sizes, delays,
      sequence_lengths, reward_densities, terminal_state_densities)


def on_train_result(info):
#     print("#############trainer.train() result: {} -> {} episodes".format(
#         info["trainer"], info["result"]["episodes_this_iter"]), info)
    # you can mutate the result dict to add new fields to return
#     stats['episode_len_mean'] = info['result']['episode_len_mean']
#     print("++++++++", aaaa, stats)
    algorithm = info["trainer"]._name
    state_space_size = info["result"]["config"]["env_config"]["state_space_size"]
    action_space_size = info["result"]["config"]["env_config"]["action_space_size"]
    delay = info["result"]["config"]["env_config"]["delay"]
    sequence_length = info["result"]["config"]["env_config"]["sequence_length"]
    reward_density = info["result"]["config"]["env_config"]["reward_density"]
    terminal_state_density = info["result"]["config"]["env_config"]["terminal_state_density"]
    fcnet_hiddens = info["result"]["config"]["model"]["fcnet_hiddens"]
    num_layers = len(fcnet_hiddens)
    layer_width = fcnet_hiddens[0] #hack

    timesteps_total = info["result"]["timesteps_total"] # also has episodes_total and training_iteration
    episode_reward_mean = info["result"]["episode_reward_mean"] # also has max and min
    episode_len_mean = info["result"]["episode_len_mean"]

    fout = open('/home/rajanr/custom-gym-env/rl_stats_temp.csv', 'a') #hardcoded
    fout.write('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, '
               'terminal_state_density, num_layers, layer_width,\n' + str(algorithm) + ' ' + str(state_space_size) +
               ' ' + str(action_space_size) + ' ' + str(delay) + ' ' + str(sequence_length)
               + ' ' + str(reward_density) + ' ' + str(terminal_state_density) + ' ')
               # Writes every iteration, would slow things down. #hack
    fout.write(str(num_layers) + ' ' + str(layer_width) + ' ' + str(timesteps_total) + ' ' + str(episode_reward_mean) +
               ' ' + str(episode_len_mean) + '\n')
    fout.close()

    info["result"]["callback_ok"] = True



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

# tune.run(
#     VIAgent,
#     stop={
#         "timesteps_total": 20000,
#           },
#     config={
#         "tolerance": 0.01,
#         "discount_factor": 0.99,
#         "rollouts_per_iteration": 10,
#       "env": "RLToy-v0",
#       "env_config": {
#         'state_space_type': 'discrete',
#         'action_space_type': 'discrete',
#         'state_space_size': 10,
#         'action_space_size': 10,
#         'generate_random_mdp': True,
#         'delay': 0,
#         'sequence_length': 1,
#         'reward_density': 0.25,
#         'terminal_state_density': 0.25
#         },
#     },
# )

for algorithm in algorithms: #TODO each one has different config_spaces
    for state_space_size in state_space_sizes:
        for action_space_size in action_space_sizes:
            for delay in delays:
                for sequence_length in sequence_lengths:
                    for reward_density in reward_densities:
                        for terminal_state_density in terminal_state_densities:
                            for num_layers in num_layerss:
                                for layer_width in layer_widths:
                                    tune.run(
                                        algorithm,
                                        stop={
                                            "timesteps_total": 20000,
                                              },
                                        config={
                                          # 'seed': 0, #seed
                                          "adam_epsilon": 0.00015,
                                          "beta_annealing_fraction": 1.0,
                                          "buffer_size": 1000000,
                                          "double_q": False,
                                          "dueling": False,
                                          "env": "RLToy-v0",
                                          "env_config": {
                                            'seed': 0, #seed
                                            'state_space_type': 'discrete',
                                            'action_space_type': 'discrete',
                                            'state_space_size': state_space_size,
                                            'action_space_size': action_space_size,
                                            'generate_random_mdp': True,
                                            'delay': delay,
                                            'sequence_length': sequence_length,
                                            'reward_density': reward_density,
                                            'terminal_state_density': terminal_state_density,
                                            'repeats_in_sequences': False,
                                            'reward_unit': 1.0,
                                            'make_denser': False,
                                            'completely_connected': True
                                            },
                                        "model": {
                                            "fcnet_hiddens": [layer_width for i in range(num_layers)],
                                            "custom_preprocessor": "ohe",
                                            "custom_options": {},  # extra options to pass to your preprocessor
                                            "fcnet_activation": "tanh",
                                            "use_lstm": False,
                                            "max_seq_len": 20,
                                            "lstm_cell_size": 256,
                                            "lstm_use_prev_action_reward": False,
                                            },
                                          "exploration_final_eps": 0.01,
                                          "exploration_fraction": 0.1,
                                          "final_prioritized_replay_beta": 1.0,
                                          "hiddens": None,
                                          "learning_starts": 2000,
                                          "lr": 6.25e-05, # "lr": grid_search([1e-2, 1e-4, 1e-6]),
                                          "n_step": 1,
                                          "noisy": False,
                                          "num_atoms": 1,
                                          "prioritized_replay": False,
                                          "prioritized_replay_alpha": 0.5,
                                          "sample_batch_size": 4,
                                          "schedule_max_timesteps": 20000,
                                          "target_network_update_freq": 80,
                                          "timesteps_per_iteration": 100,
                                          "train_batch_size": 32,

                                                  "callbacks": {
                                    #                 "on_episode_start": tune.function(on_episode_start),
                                    #                 "on_episode_step": tune.function(on_episode_step),
                                    #                 "on_episode_end": tune.function(on_episode_end),
                                    #                 "on_sample_end": tune.function(on_sample_end),
                                                    "on_train_result": tune.function(on_train_result),
                                    #                 "on_postprocess_traj": tune.function(on_postprocess_traj),
                                                },
                                        },
                                     #return_trials=True # add tirals = tune.run( above
                                     )

end = time.time()
print("No. of seconds to run:", end - start)
