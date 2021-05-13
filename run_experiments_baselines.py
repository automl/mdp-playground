'''Script to run experiments on MDP Playground.

Takes a configuration file, experiment name and config number to run as optional arguments.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

import mdp_playground
from mdp_playground.envs import RLToyEnv

import sys, os
import argparse
import importlib

import default_config
print("default_config:", default_config)

import gym
from gym.wrappers.time_limit import TimeLimit
#from gym.wrappers.monitor import Monitor

import stable_baselines as sb
from stable_baselines import DQN, DDPG, SAC, A2C, TD3
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import time
import pprint
from mdp_playground.config_processor.baselines_processor import *


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


def process_dictionaries(config, current_config, var_env_configs, var_agent_configs, var_model_configs):
    algorithm = config.algorithm
    agent_config = config.agent_config
    model_config = config.model_config
    env_config = config.env_config

    for config_type, config_dict in config.var_configs.items():
        for key in config_dict:
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
                # if algorithm == 'SAC' and key == 'learning_rate': #hack
                #     pass
                #     value = current_config[num_configs_done + list(config.var_configs[config_type]).index(key)]
                #     agent_config['optimization'] = {
                #                                     key: value,
                #                                     'actor_learning_rate': value,
                #                                     'entropy_learning_rate': value,
                #                                     }
                # elif algorithm == 'SAC' and key == 'layers': #hack
                #     agent_config['Q_model'] = {
                #                                 key: current_config[num_configs_done + list(config.var_configs[config_type]).index(key)],
                #                                 "fcnet_activation": "relu",
                #                                 }
                #     agent_config['policy_model'] = {
                #                                 key: current_config[num_configs_done + list(config.var_configs[config_type]).index(key)],
                #                                 "fcnet_activation": "relu",
                #                                 }
                # else:
                agent_config[key] = current_config[num_configs_done + list(config.var_configs[config_type]).index(key)]

            elif config_type == "model":
                num_configs_done = len(list(var_env_configs)) + len(list(var_agent_configs))
                model_config["model"][key] = current_config[num_configs_done + list(config.var_configs[config_type]).index(key)]

    #hacks begin:
    # if "use_lstm" in model_config["model"]:
    #     if model_config["model"]["use_lstm"]:#if true
    #         model_config["model"]["max_seq_len"] = env_config["env_config"]["delay"] + env_config["env_config"]["sequence_length"] + 1

    if algorithm == 'TD3':
        if("target_policy_noise" in agent_config):
            agent_config["target_noise_clip"] = agent_config["target_noise_clip"] * agent_config["target_policy_noise"]
    if("state_space_type" in env_config["env_config"]):
        if env_config["env_config"]["state_space_type"] == 'continuous':
            env_config["env_config"]["action_space_dim"] = env_config["env_config"]["state_space_dim"]
    
    return agent_config, model_config, env_config


def main(args):   
    # print("Config file", os.path.abspath(args.config_file)) # 'experiments/dqn_seq_del.py'
    if args.config_file[-3:] == '.py':
        args.config_file = args.config_file[:-3]

    config_file_path = os.path.abspath('/'.join(args.config_file.split('/')[:-1]))
    print("config_file_path:", config_file_path)
    
    sys.path.insert(1, config_file_path) #hack
    config = importlib.import_module(args.config_file.split('/')[-1], package=None)
    print("Number of seeds for environment:", config.num_seeds)
    print("Configuration numbers that will be run:", "all" if args.config_num is None else args.config_num)

    args.exp_name = os.path.abspath(args.exp_name)
    if args.config_num is None:
        stats_file_name = args.exp_name
    else:
        stats_file_name = args.exp_name + '_' + str(args.config_num)
    print("Stats file being written to:", stats_file_name)

    # ------------------------- Variable configurations  ----------------------------#
    config, var_agent_configs, var_model_configs = agent_to_baselines(config)
    var_configs_deepcopy = copy.deepcopy(config.var_configs) #hack because this needs to be read in on_train_result and trying to read config there raises an error because it's been imported from a Python module and I think they try to reload it there.

    if "env" in config.var_configs:
        var_env_configs = config.var_configs["env"] #hack
    else:
        var_env_configs = []
    config_algorithm = config.algorithm #hack, used on on_train_result

    print('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, make_denser, terminal_state_density, transition_noise, reward_noise ')

    configs_to_print = ''
    for config_type, config_dict in var_configs_deepcopy.items():
        if config_type == 'env':
            for key in config_dict:
                configs_to_print += str(config_dict[key]) + ', '

    print(config_algorithm, configs_to_print)

    #----------------- Write headers to csv file --------------------#
    hack_filename = stats_file_name + '.csv'
    fout = open(hack_filename, 'a') #hardcoded
    fout.write('# training_iteration, algorithm, ')
    for config_type, config_dict in var_configs_deepcopy.items():
        for key in config_dict:
            fout.write(key + ', ')
    fout.write('timesteps_total, episode_reward_mean, episode_len_mean\n')
    fout.close()

    #---------- Compute cartesian product of configs ----------------#
    start = time.time()
    if args.config_num is None:
        cartesian_product_configs = config.final_configs
    else:
        cartesian_product_configs = [config.final_configs[args.config_num]]


    # ------------ Run every configuration  ------------------ #
    pp = pprint.PrettyPrinter(indent=4)
    for i, current_config in enumerate(cartesian_product_configs):
        print("Configuration: %d \t %d remaining"
              % (i, len(cartesian_product_configs) - i))
        # ------------ Dictionaries setup  ------------------ #
        agent_config, model_config, env_config = \
            process_dictionaries(config, current_config, var_env_configs, var_agent_configs, var_model_configs)  # hacks

        # ------------ Env init  ------------------ #
        env = gym.make(current_config['env'],
                       **current_config['env_config'])

        # if time limit, wrap it
        if("horizon" in current_config['env']):
            train_horizon = current_config['env']["horizon"]
            env = TimeLimit(env, max_episode_steps=train_horizon)  # horizon
        env = sb.bench.Monitor(env, None)  # , "../", force=True)
        # Automatically normalize the input features and reward
        if (config_algorithm == "DDPG"
           or config_algorithm == "A2C"
           or config_algorithm == "A3C"):
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(env,
                               norm_obs=False,
                               norm_reward=True,
                               clip_obs=10.)

        # limit max timesteps in evaluation
        eval_horizon = 100
        eval_env = gym.make(env_config['env'], **env_config['env_config'])
        eval_env = TimeLimit(eval_env, max_episode_steps=eval_horizon)
        print("Env config:",)
        pp.pprint(env_config)

        #------------ Agent init  ------------------ #
        # Create the agent
        timesteps_per_iteration = current_config["timesteps_per_iteration"]
        use_lstm = current_config["use_lstm"]
        agent_config_baselines = current_config["agent"]
        model = init_agent(env,
                           config_algorithm,
                           agent_config_baselines, use_lstm)

        #------------ train/evaluation ------------------ #
        timesteps_total = config.timesteps_total

        # train your model for n_iter timesteps
        #Define evaluation
        eval_interval = config.evaluation_interval  # every x training iterations
        csv_callbacks = CustomCallback(eval_env,
                                       n_eval_episodes=10,
                                       timesteps_per_iteration=timesteps_per_iteration,
                                       eval_interval=eval_interval,
                                       deterministic=True,
                                       file_name=stats_file_name,
                                       config_algorithm=config_algorithm,
                                       var_configs=var_configs_deepcopy)
        #Train
        learn_params = {"callback": csv_callbacks,
                        "total_timesteps": int(timesteps_total)}
        if(config_algorithm == "DDPG"): #Log interval is handled differently for each algorithm, e.g. each log_interval episodes(DQN) or log_interval steps(DDPG).
            learn_params["log_interval"] = timesteps_per_iteration
        else:
            learn_params["log_interval"] = timesteps_per_iteration//10
        
        model.learn(**learn_params)
        model.save('%s_last'%(args.exp_name))

    end = time.time()
    print("No. of seconds to run:", end - start)

if __name__ == '__main__':
    args = parse_args()
    main(args)