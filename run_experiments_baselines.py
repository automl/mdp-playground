'''Script to run experiments on MDP Playground.

Takes a configuration file, experiment name and config number to run as optional arguments.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing_extensions import final

import numpy as np
import copy

import mdp_playground
from mdp_playground.envs import RLToyEnv

import sys, os
import argparse
import default_config
print("default_config:", default_config)

import gym
from gym.wrappers.time_limit import TimeLimit

import time
import pprint
import mdp_playground.config_processor.config_processor as config_processor
from mdp_playground.config_processor.baselines_processor import *
import logging
log_level_ = logging.WARNING

def parse_args():
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
    args = parser.parse_args()
    print("Parsed args:", args)
    return args


def main(args):   
    # print("Config file", os.path.abspath(args.config_file)) # 'experiments/dqn_seq_del.py'
    if args.config_file[-3:] == '.py':
        config_file = args.config_file[:-3]
    else:
        config_file = args.config_file

    stats_file_name = os.path.abspath(args.exp_name)
    if args.config_num is None:
        stats_file_name = args.exp_name
    else:
        stats_file_name = args.exp_name + '_' + str(args.config_num)
    print("Stats file being written to:", stats_file_name)

    # config_file_path = os.path.abspath('/'.join(args.config_file.split('/')[:-1]))
    # print("config_file_path:", config_file_path)

    # sys.path.insert(1, config_file_path)  # hack
    # config = importlib.import_module(args.config_file.split('/')[-1], package=None)
    # print("Number of seeds for environment:", config.num_seeds)
    # print("Configuration numbers that will be run:", "all" if args.config_num is None else args.config_num)

    config, final_configs = \
        config_processor.process_configs(config_file,
                                         stats_file_prefix=stats_file_name,
                                         framework="stable_baselines",
                                         config_num=args.config_num,
                                         log_level=log_level_)
    # ------------------------- Variable configurations  ----------------------------#
    print("Configuration number(s) that will be run:", "all" if args.config_num is None else args.config_num)


    # ------------ Run every configuration  ------------------ #
    pp = pprint.PrettyPrinter(indent=4)
    total_configs = len(final_configs)
    config_algorithm = config.algorithm
    eval_cfg = config.eval_config
    for i, current_config in enumerate(final_configs):
        print("Configuration: %d \t %d remaining"
              % (i, total_configs - i))

        # ------------ Env init  ------------------ #
        env_config = current_config['env_config']
        print("Env config:",)
        pp.pprint(env_config)
        env, eval_env = \
            init_environments(config_algorithm,
                              current_config,
                              env_config,
                              eval_cfg["evaluation_config"])
        # ------------ Agent init  ------------------ #
        # Create the agent
        timesteps_per_iteration = current_config["timesteps_per_iteration"]
        model = init_agent(env,
                           config_algorithm,
                           current_config["model"],
                           current_config["agent"])

        # ------------ train/evaluation ------------------ #
        timesteps_total = config.timesteps_total

        # train your model for n_iter timesteps
        # Define evaluation
        # every x training iterations
        eval_interval = eval_cfg["evaluation_interval"]
        stats_file_name = current_config["stats_file_name"]
        var_configs_deepcopy = current_config["var_configs_deepcopy"]
        csv_callbacks = CustomCallback(eval_env,
                                       n_eval_episodes=10,
                                       timesteps_per_iteration=timesteps_per_iteration,
                                       eval_interval=eval_interval,
                                       deterministic=True,
                                       file_name=stats_file_name,
                                       config_algorithm=config_algorithm,
                                       var_configs=var_configs_deepcopy)
        # Train
        learn_params = {"callback": csv_callbacks,
                        "total_timesteps": int(timesteps_total)}
        if(config_algorithm == "DDPG"):
            # Log interval is handled differently for each algorithm,
            # e.g. each log_interval episodes(DQN) or log_interval steps(DDPG).
            learn_params["log_interval"] = timesteps_per_iteration
        else:
            learn_params["log_interval"] = timesteps_per_iteration//10

        model.learn(**learn_params)
        model.save('%s_last' % (args.exp_name))

    end = time.time()
    print("No. of seconds to run:", end - start)


if __name__ == '__main__':
    args = parse_args()
    main(args)
