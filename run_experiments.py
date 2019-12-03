from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.utils.annotations import override
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.offline import OutputWriter


import ray
from ray import tune
from ray.rllib.utils.seed import seed as rllib_seed
import rl_toy
from rl_toy.envs import RLToyEnv
from ray.tune.registry import register_env
register_env("RLToy-v0", lambda config: RLToyEnv(config))

import sys, os
print("Arguments", sys.argv) # Ray problems: writing to output file (could be fixed with IOContext?); fine-grained control over NN arch.; random seeds; env steps per algo.)
import argparse
import configparser

parser = argparse.ArgumentParser(description='Run experiments for MDP Playground')
parser.add_argument('--config-file', dest='config_file', action='store', default='default_config',
                   help='Configuration file containing configuration space to run. It must be a Python file so config can be given programmatically. '
                   'Remove the .py extension when providing the filename. See default_config.py for an example.')
parser.add_argument('--output-file', dest='csv_stats_file', action='store', default='temp234',
                   help='Prefix of output file. It will save stats to 2 CSV files, with the filenames as the one given as argument'
                   'and another file with an extra "_eval" in the filename that contains evaluation stats during the training. Appends to existing files or creates new ones if they don\'t exist.')

args = parser.parse_args()
print("Parsed args:", args)

import importlib
config = importlib.import_module(args.config_file, package=None)
print(config.num_seeds)

# Did not do ConfigParser or JSON because I want to give config programmatically
# config = configparser.ConfigParser()
# config.read(args.config_file)
# print(config.sections(), [i for i in config['ConfigSpace']])

# import json
# print([json.loads(config['ConfigSpace'][i]) for i in config['ConfigSpace']])

from ray.rllib.models.preprocessors import OneHotPreprocessor
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_preprocessor("ohe", OneHotPreprocessor)


ray.init(local_mode=True)#, object_id_seed=0)


# num_seeds = 10
# state_space_sizes = [8]#, 10, 12, 14] # [2**i for i in range(1,6)]
# action_space_sizes = [8]#2, 4, 8, 16] # [2**i for i in range(1,6)]
# delays = [0] + [2**i for i in range(4)]
# sequence_lengths = [1, 2, 3, 4]#i for i in range(1,4)]
# reward_densities = [0.25] # np.linspace(0.0, 1.0, num=5)
# # make_reward_dense = [True, False]
# terminal_state_densities = [0.25] # np.linspace(0.1, 1.0, num=5)
# algorithms = ["DQN"]
# seeds = [i for i in range(num_seeds)]
# # Others, keep the rest fixed for these: learning_starts, target_network_update_freq, double_dqn, fcnet_hiddens, fcnet_activation, use_lstm, lstm_seq_len, sample_batch_size/train_batch_size, learning rate
# # More others: adam_epsilon, exploration_final_eps/exploration_fraction, buffer_size
# num_layerss = [1, 2, 3, 4]
# layer_widths = [8, 32, 128]
# fcnet_activations = ["tanh", "relu", "sigmoid"]
# learning_startss = [500, 1000, 2000, 4000, 8000]
# target_network_update_freqs = [8, 80, 800]
# double_dqn = [False, True]
# learning_rates = []

# lstm with sequence lengths

print('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, terminal_state_density ')
print(config.algorithms, config.state_space_sizes, config.action_space_sizes, config.delays, config.sequence_lengths, config.reward_densities, config.terminal_state_densities)



#TODO Write addnl. line at beginning of file for column names for eval file
# fout = open('rl_stats_temp.csv', 'a') #hardcoded
# fout.write('# basename, n_points, n_features, n_trees ')

hack_filename = args.csv_stats_file + '.csv'
fout = open(hack_filename, 'a') #hardcoded
fout.write('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, terminal_state_density, dummy_seed,\n')
fout.close()

# sys.exit(0)

import time
start = time.time()


def on_train_result(info):
#     print("#############trainer.train() result: {} -> {} episodes".format(
#         info["trainer"], info["result"]["episodes_this_iter"]), info)
    # you can mutate the result dict to add new fields to return
#     stats['episode_len_mean'] = info['result']['episode_len_mean']
#     print("++++++++", aaaa, stats)
    training_iteration = info["result"]["training_iteration"]
    algorithm = info["trainer"]._name
    state_space_size = info["result"]["config"]["env_config"]["state_space_size"]
    action_space_size = info["result"]["config"]["env_config"]["action_space_size"]
    delay = info["result"]["config"]["env_config"]["delay"]
    sequence_length = info["result"]["config"]["env_config"]["sequence_length"]
    reward_density = info["result"]["config"]["env_config"]["reward_density"]
    terminal_state_density = info["result"]["config"]["env_config"]["terminal_state_density"]
    dummy_seed = info["result"]["config"]["env_config"]["dummy_seed"]

    timesteps_total = info["result"]["timesteps_total"] # also has episodes_total and training_iteration
    episode_reward_mean = info["result"]["episode_reward_mean"] # also has max and min
    episode_len_mean = info["result"]["episode_len_mean"]

    fout = open(hack_filename, 'a') #hardcoded
    fout.write(str(training_iteration) + ' ')
    fout.write(str(algorithm) + ' ' + str(state_space_size) +
               ' ' + str(action_space_size) + ' ' + str(delay) + ' ' + str(sequence_length)
               + ' ' + str(reward_density) + ' ' + str(terminal_state_density) + ' ')
               # Writes every iteration, would slow things down. #hack
    fout.write(str(dummy_seed) + ' ' + str(timesteps_total) + ' ' + str(episode_reward_mean) +
               ' ' + str(episode_len_mean) + '\n')
    fout.close()

    # print("###HACK info object:", info)
    hack_filename_eval = args.csv_stats_file + '_eval.csv'
    fout = open(hack_filename_eval, 'a') #hardcoded
    fout.write('#HACK STRING EVAL' + "\n")
    fout.close()

    info["result"]["callback_ok"] = True


def on_episode_end(info):
    # if not info["env"].config["make_denser"]:
#    print("###on_episode_end", info["episode"].agent_rewards)

    #info has env, policy, Episode objects
    if "dummy_eval" in info["env"].get_unwrapped()[0].config:
        print("###on_episode_end info", info["env"].get_unwrapped()[0].config["make_denser"], info["episode"].total_reward, info["episode"].length) #, info["episode"]._agent_reward_history)
        reward_this_episode = info["episode"].total_reward
        length_this_episode = info["episode"].length
        hack_filename_eval = args.csv_stats_file + '_eval.csv'
        fout = open(hack_filename_eval, 'a') #hardcoded
        fout.write(str(reward_this_episode) + ' ' + str(length_this_episode) + "\n")
        fout.close()


for algorithm in config.algorithms: #TODO each one has different config_spaces
    for state_space_size in config.state_space_sizes:
        for action_space_size in config.action_space_sizes:
            for delay in config.delays:
                for sequence_length in config.sequence_lengths:
                    for reward_density in config.reward_densities:
                        for terminal_state_density in config.terminal_state_densities:
                            for dummy_seed in config.seeds: #TODO Different seeds for Ray Trainer (TF, numpy, Python; Torch, Env), Environment (it has multiple sources of randomness too), Ray Evaluator
                                tune.run(
                                    algorithm,
                                    stop={
                                        "timesteps_total": 20000,
                                          },
                                    config={
                                      "adam_epsilon": 1e-4,
                                      "beta_annealing_fraction": 1.0,
                                      "buffer_size": 1000000,
                                      "double_q": False,
                                      "dueling": False,
                                      "exploration_final_eps": 0.01,
                                      "exploration_fraction": 0.1,
                                      "final_prioritized_replay_beta": 1.0,
                                      "hiddens": None,
                                      "learning_starts": 1000,
                                      "lr": 1e-4, # "lr": grid_search([1e-2, 1e-4, 1e-6]),
                                      "n_step": 1,
                                      "noisy": False,
                                      "num_atoms": 1,
                                      "prioritized_replay": False,
                                      "prioritized_replay_alpha": 0.5,
                                      "sample_batch_size": 4,
                                      "schedule_max_timesteps": 20000,
                                      "target_network_update_freq": 800,
                                      "timesteps_per_iteration": 100,
                                      "train_batch_size": 32,

                                      "env": "RLToy-v0",
                                      "env_config": {
                                        'dummy_seed': dummy_seed,
                                        'seed': 1, #seed
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
                                        "fcnet_hiddens": [256, 256],
                                        "custom_preprocessor": "ohe",
                                        "custom_options": {},  # extra options to pass to your preprocessor
                                        "fcnet_activation": "tanh",
                                        "use_lstm": False,
                                        "max_seq_len": 20,
                                        "lstm_cell_size": 256,
                                        "lstm_use_prev_action_reward": False,
                                        },

                                              "callbacks": {
                                #                 "on_episode_start": tune.function(on_episode_start),
                                #                 "on_episode_step": tune.function(on_episode_step),
                                                "on_episode_end": tune.function(on_episode_end),
                                #                 "on_sample_end": tune.function(on_sample_end),
                                                "on_train_result": tune.function(on_train_result),
                                #                 "on_postprocess_traj": tune.function(on_postprocess_traj),
                                            },
                                        "evaluation_interval": 1, # I think this every x training_iterations
                                        "evaluation_config": {
                                        "exploration_fraction": 0,
                                        "exploration_final_eps": 0,
                                        "batch_mode": "complete_episodes",
                                        'horizon': 100,
                                          "env_config": {
                                            "dummy_eval": True, #hack
                                            }
                                    },
                                    },
                                 #return_trials=True # add trials = tune.run( above
                                 )

end = time.time()
print("No. of seconds to run:", end - start)
