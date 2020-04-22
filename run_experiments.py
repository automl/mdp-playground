from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import ray
from ray import tune
from ray.rllib.utils.seed import seed as rllib_seed
import mdp_playground
from mdp_playground.envs import RLToyEnv
from ray.tune.registry import register_env
register_env("RLToy-v0", lambda config: RLToyEnv(config))

import sys, os
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
print("Number of seeds for environment:", config.num_seeds)


from ray.rllib.models.preprocessors import OneHotPreprocessor
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_preprocessor("ohe", OneHotPreprocessor)

ray.init(local_mode=True)

print('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, terminal_state_density ')
print(config.algorithms, config.state_space_sizes, config.action_space_sizes, config.delays, config.sequence_lengths, config.reward_densities, config.terminal_state_densities)


hack_filename = args.csv_stats_file + '.csv'
fout = open(hack_filename, 'a') #hardcoded
fout.write('# training_iteration, Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, terminal_state_density, transition_noise, reward_noise, dummy_seed, timesteps_total, episode_reward_mean, episode_len_mean\n')
fout.close()

# sys.exit(0)

import time
start = time.time()

# Ray callback to write training stats to CSV file at end of every training iteration
def on_train_result(info):
    training_iteration = info["result"]["training_iteration"]
    algorithm = info["trainer"]._name
    state_space_size = info["result"]["config"]["env_config"]["state_space_size"]
    action_space_size = info["result"]["config"]["env_config"]["action_space_size"]
    delay = info["result"]["config"]["env_config"]["delay"]
    sequence_length = info["result"]["config"]["env_config"]["sequence_length"]
    reward_density = info["result"]["config"]["env_config"]["reward_density"]
    terminal_state_density = info["result"]["config"]["env_config"]["terminal_state_density"]
    dummy_seed = info["result"]["config"]["env_config"]["dummy_seed"]
    transition_noise = info["result"]["config"]["env_config"]["transition_noise"]
    reward_noise = info["result"]["config"]["env_config"]["reward_noise_std"]

    timesteps_total = info["result"]["timesteps_total"] # also has episodes_total and training_iteration
    episode_reward_mean = info["result"]["episode_reward_mean"] # also has max and min
    episode_len_mean = info["result"]["episode_len_mean"]

    fout = open(hack_filename, 'a') #hardcoded
    fout.write(str(training_iteration) + ' ')
    fout.write(str(algorithm) + ' ' + str(state_space_size) +
               ' ' + str(action_space_size) + ' ' + str(delay) + ' ' + str(sequence_length)
               + ' ' + str(reward_density) + ' ' + str(terminal_state_density) + ' ')
               # Writes every iteration, would slow things down. #hack
    fout.write(str(transition_noise) + ' ' + str(reward_noise) + ' ' + str(dummy_seed) + ' ' + str(timesteps_total) + ' ' + str(episode_reward_mean) +
               ' ' + str(episode_len_mean) + '\n')
    fout.close()

    # We did not manage to find an easy way to log evaluation stats for Ray without the following hack which demarcates the end of a training iteration in the evaluation stats file
    hack_filename_eval = args.csv_stats_file + '_eval.csv'
    fout = open(hack_filename_eval, 'a') #hardcoded
    fout.write('#HACK STRING EVAL' + "\n")
    fout.close()

    info["result"]["callback_ok"] = True


# Ray callback to write evaluation stats to CSV file at end of every training iteration
def on_episode_end(info):
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
                            for transition_noise in config.transition_noises:
                                for reward_noise in config.reward_noises:
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
                                                'completely_connected': True,
                                                'transition_noise': transition_noise,
                                                'reward_noise': tune.function(lambda a: a.normal(0, reward_noise)),
                                                # 'reward_noise_std': reward_noise,
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
                                            "evaluation_interval": 1, # I think this means every x training_iterations
                                            "evaluation_config": {
                                            "exploration_fraction": 0,
                                            "exploration_final_eps": 0,
                                            "batch_mode": "complete_episodes",
                                            'horizon': 100,
                                              "env_config": {
                                                "dummy_eval": True, #hack
                                                'transition_noise': 0,
                                                'reward_noise': tune.function(lambda a: a.normal(0, 0))
                                                }
                                            },
                                            },
                                         #return_trials=True # add trials = tune.run( above
                                         )

end = time.time()
print("No. of seconds to run:", end - start)
