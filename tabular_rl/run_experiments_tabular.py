'''Script to run experiments on MDP Playground.

Takes a configuration file, experiment name and config number to run as optional arguments.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
# sys.path.insert(1, "./mdp_playground/envs/")
sys.path.append('/home/rajanr/mdp-playground/tabular_rl/')


import numpy as np
import copy
import random
import pandas as pd
from tabular_rl.agents.Q_learning import q_learning
from tabular_rl.agents.Double_Q_learning import double_q_learning
from tabular_rl.agents.Sarsa import sarsa

import mdp_playground
from mdp_playground.envs import RLToyEnv

import argparse
from pathlib import Path
from datetime import datetime

from collections import OrderedDict
# import configparser

parser = argparse.ArgumentParser(description=__doc__) # docstring at beginning of the file is stored in __doc__
parser.add_argument('-c', '--config-file', dest='config_file', action='store', default='default_config',
                   help='Configuration file containing configuration to run experiments. It must be a Python file so config can be given programmatically. There are 2 types of configs - VARIABLE CONFIG across the experiments and STATIC CONFIG across the experiments. \nVARIABLE CONFIGS: The OrderedDicts var_env_configs, var_agent_configs and var_model_configs hold configuration options that are variable for the environment, agent and model across the current experiment. For each configuration option, the option is the key in the dict and its value is a list of values it can take for the current experiment.  A Cartesian product of these lists is taken to generate various possible configurations to be run. For example, you might want to vary "delay" for the current experiment. Then "delay" would be a key in var_env_configs dict and its value would be a list of values it can take. Because Ray does not have a common way to address this specification of configurations for its agents, there are a few hacky ways to set var_agent_configs and var_model_configs currently. Please see sample experiment config files in the experiments directory to see how to set the values for a given algorithm. \nSTATIC CONFIGS: env_config, agent_config and model_config are dicts which hold the static configuration for the current experiment as a normal Python dict.')
parser.add_argument('-e', '--exp-name', dest='exp_name', action='store', default='mdpp_default_experiment',
                   help='The user-chosen name of the experiment. This is used as the prefix of the output files (the prefix also contains config_num if that is provided). It will save stats to 2 CSV files, with the filenames as the one given as argument'
                   ' and another file with an extra "_eval" in the filename that contains evaluation stats during the training. Appends to existing files or creates new ones if they don\'t exist.')
parser.add_argument('-n', '--config-num', dest='config_num', action='store', default=None, type=int,
                   help='Used for running the configurations of experiments in parallel. This is appended to the prefix of the output files (after exp_name).'
                   ' A Cartesian product of different configuration values for the experiment will be taken and ordered as a list and this number corresponds to the configuration number in this list.'
                   ' Please look in to the code for details.')
# parser.add_argument('-t', '--tune-hps', dest='tune_hps', action='store', default=False, type=bool,
#                    help='Used for tuning the hyperparameters that can be used for experiments later.'
#                    ' A Cartesian product of different configuration values for the experiment will be taken and ordered as a list and this number corresponds to the configuration number in this list.'
#                    ' Please look in to the code for details.')


args = parser.parse_args()
print("Parsed args:", args)

if args.config_file[-3:] == '.py':
    args.config_file = args.config_file[:-3]

config_file_path = os.path.abspath('/'.join(args.config_file.split('/')[:-1]))
# print("config_file_path:", config_file_path)
sys.path.insert(1, config_file_path) #hack
import importlib
config = importlib.import_module(args.config_file.split('/')[-1], package=None)
print("Number of seeds for environment:", config.num_seeds)
print("Configuration numbers that will be run:", "all" if args.config_num is None else args.config_num)


# import default_config
# print("default_config:", default_config)
# print(os.path.abspath(args.config_file)) # 'experiments/dqn_seq_del.py'

args.exp_name = os.path.abspath(args.exp_name)
if args.config_num is None:
    stats_file_name = args.exp_name
else:
    stats_file_name = args.exp_name + '_' + str(args.config_num)
print("Stats file being written to:", stats_file_name)


#TODO Different seeds for Ray Trainer (TF, numpy, Python; Torch, Env), Environment (it has multiple sources of randomness too), Ray Evaluator


var_configs_deepcopy = copy.deepcopy(config.var_configs) #hack because this needs to be read in on_train_result and trying to read config there raises an error because it's been imported from a Python module and I think they try to reload it there.

if "env" in config.var_configs:
    var_env_configs = config.var_configs["env"] #hack
else:
    var_env_configs = []
if "agent" in config.var_configs:
    var_agent_configs = config.var_configs["agent"] #hack
else:
    var_agent_configs = []
if "model" in config.var_configs:
    var_model_configs = config.var_configs["model"] #hack
else:
    var_model_configs = []

config_algorithm = config.algorithm #hack
# sys.exit(0)


print('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, make_denser, terminal_state_density, transition_noise, reward_noise ')

configs_to_print = ''
for config_type, config_dict in var_configs_deepcopy.items():
    if config_type == 'env':
        for key in config_dict:
            configs_to_print += str(config_dict[key]) + ', '

print(config.algorithm, configs_to_print)

import time
start = time.time()
value_tuples = []
for config_type, config_dict in config.var_configs.items():
    for key in config_dict:
        assert type(config.var_configs[config_type][key]) == list, "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
        value_tuples.append(config.var_configs[config_type][key])

import itertools
cartesian_product_configs = list(itertools.product(*value_tuples))
print("Total number of configs. to run:", len(cartesian_product_configs))

if args.config_num is None:
    pass
else:
    print("Current config to run:", cartesian_product_configs[args.config_num])
    cartesian_product_configs = [cartesian_product_configs[args.config_num]]

import pprint
pp = pprint.PrettyPrinter(indent=4)

# time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#dir_name = time + "_" + "q_learn_tabular_del" + "_" + str(args.exp_name.split('/')[-1]) + '_' + str(args.config_num)
# dir_name =
#
# path = Path("experiments") / dir_name
# if not os.path.exists(path):
#     os.makedirs(path)

for current_config in cartesian_product_configs:
    algorithm = config.algorithm

    agent_config = config.agent_config
    model_config = config.model_config
    env_config = config.env_config
    # sys.exit(0)

    for config_type, config_dict in config.var_configs.items():
        for key in config_dict:
        # if config_type == "env_config": # There is a dummy seed in the env_config because it's not used in the environment. It implies a different seed for the agent on every launch as the seed for Ray is not being set here. I faced problems with Ray's seeding process.
            if config_type == "env":
                if key == 'reward_noise':
                    reward_noise_ = current_config[list(var_env_configs).index(key)] # this works because env_configs are 1st in the OrderedDict
                    env_config["env_config"][key] = lambda a: a.normal(0, reward_noise_)
                    env_config["env_config"]['reward_noise_std'] = reward_noise_ #hack Needed to be able to write scalar value of std dev. to stats file instead of the lambda function above ###TODO Could remove the hack by creating a class for the noises and changing its repr()
                elif key == 'transition_noise' and env_config["env_config"]["state_space_type"] == "continuous":
                    transition_noise_ = current_config[list(var_env_configs).index(key)]
                    env_config["env_config"][key] = lambda a: a.normal(0, transition_noise_)
                    env_config["env_config"]['transition_noise_std'] = transition_noise_ #hack
                else:
                    env_config["env_config"][key] = current_config[list(var_env_configs).index(key)]

            elif config_type == "agent":
                num_configs_done = len(list(var_env_configs))
                if algorithm == 'SAC' and key == 'critic_learning_rate': #hack
                    value = current_config[num_configs_done + list(config.var_configs[config_type]).index(key)]
                    agent_config['optimization'] = {
                                                    key: value,
                                                    'actor_learning_rate': value,
                                                    'entropy_learning_rate': value,
                                                    }
                elif algorithm == 'SAC' and key == 'fcnet_hiddens': #hack
                    agent_config['Q_model'] = {
                                                key: current_config[num_configs_done + list(config.var_configs[config_type]).index(key)],
                                                "fcnet_activation": "relu",
                                                }
                    agent_config['policy_model'] = {
                                                key: current_config[num_configs_done + list(config.var_configs[config_type]).index(key)],
                                                "fcnet_activation": "relu",
                                                }
                else:
                    agent_config[key] = current_config[num_configs_done + list(config.var_configs[config_type]).index(key)]

            elif config_type == "model":
                num_configs_done = len(list(var_env_configs)) + len(list(var_agent_configs))
                model_config["model"][key] = current_config[num_configs_done + list(config.var_configs[config_type]).index(key)]

    if "state_space_type" in env_config:
        if env_config["env_config"]["state_space_type"] == 'continuous':
            env_config["env_config"]["action_space_dim"] = env_config["env_config"]["state_space_dim"]

    #eval_config = config.eval_config

    # all_configs = {**agent_config, **model_config, **env_config, **eval_config} # This works because the dictionaries involved have mutually exclusive sets of keys, otherwise we would need to use a deepmerge!
    # print("total config:",)
    # pp.pprint(all_configs)

    agent_config["timesteps_total"] = 20000
    agent_config["horizon"] = 100
    env = RLToyEnv(**env_config["env_config"])

    print("Env config:", )
    pp.pprint(env_config)

    print("\n\033[1;32m======== Running on environment: " + env_config["env"] + " =========\033[0;0m\n")

    if isinstance(var_agent_configs, OrderedDict):
        all_variable_configs_dct = OrderedDict({**var_env_configs, **var_agent_configs})
    else:
        all_variable_configs_dct = OrderedDict(var_env_configs)

    # order of variable configs and value_tuples need to match here
    location_dummy_seed = list(all_variable_configs_dct.keys()).index("dummy_seed")
    if location_dummy_seed is None:
        raise ValueError("dummy seed needs to be set")

    print("setting the seed to {}".format(current_config[location_dummy_seed]))
    np.random.seed(current_config[location_dummy_seed])
    random.seed(current_config[location_dummy_seed])

    print("agent_name:", config.agent_name)
    print("agent_config", agent_config)

    if "double_q_learn" in algorithm:
        train_data, test_data, num_steps, timesteps_per_iteration_statistics = double_q_learning(env, **agent_config)
    elif "q_learn" in algorithm:
        train_data, test_data, num_steps, timesteps_per_iteration_statistics = q_learning(env, **agent_config)
    elif "sarsa" in algorithm:
        train_data, test_data, num_steps, timesteps_per_iteration_statistics = sarsa(env, **agent_config)

    first_keys = {"#training_iteration,": [], "algorithm,": []}

    #keys_to_exclude_from_middle_dict = ["reward_noise"]
    middle_keys = {}
    # order of variable configs and value_tuples need to match here also
    assert len(all_variable_configs_dct) == len(current_config)
    for i, kv in enumerate(all_variable_configs_dct.items()):
        key, value = kv
        if len(value) > 1:
            middle_keys[key + ","] = [current_config[i]] * len(timesteps_per_iteration_statistics)
        else:
            middle_keys[key + ","] = value * len(timesteps_per_iteration_statistics)

    last_keys = {"timesteps_total,": [], "episode_reward_mean,": [], "episode_len_mean": []}

    for i, stat in enumerate(timesteps_per_iteration_statistics, 1):
        first_keys["#training_iteration,"].append(i)
        first_keys["algorithm,"].append(algorithm)

        last_keys["timesteps_total,"].append(stat[0])
        last_keys["episode_reward_mean,"].append(stat[1])
        last_keys["episode_len_mean"].append(stat[2])


# transition_noise, dummy_seed, alpha, epsilon, epsilon_decay, timesteps_total,
    data = OrderedDict(**first_keys, **middle_keys, **last_keys)

    log_df = pd.DataFrame(data)
    log_df.to_csv(algorithm + ".csv", mode="a", sep=" ", index=False)

end = time.time()
print("No. of seconds to run:", end - start)
