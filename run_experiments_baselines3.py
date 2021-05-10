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

import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN, DDPG, SAC, A2C, TD3
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from typing import Any, Dict, List, Optional, Union

from functools import reduce
import time
import itertools
import pprint

#-------------- Custom Classes ------------#
class VisionNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, **kwargs):
        super(VisionNet, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]

        if ("act_fun" in kwargs):
            activ_fn = kwargs['activation_fun']
        else: #default
            activ_fn = nn.ReLU

        #Create Cnn
        modules = []
        for i, layer_def in enumerate(kwargs['cnn_layers']):
            n_filters, kernel, padding = layer_def
            modules.append(nn.Conv2d(n_input_channels, n_filters, kernel_size =  kernel, stride = 1, padding = padding))
            modules.append(activ_fn())
        modules.append(nn.Flatten())
        self.cnn = nn.Sequential(*modules)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).permute(0,-1, 1, 2).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), activ_fn)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.permute(0,-1, 1, 2)))

class CustomCallback(BaseCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param deterministic: (bool) Whether the evaluation should use a stochastic or deterministic actions.
    :param verbose: (int)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 1,
        file_name: str = "./evaluation.csv",
        config_algorithm: str =""
    ):
        super(CustomCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.file_name = file_name
        self.training_iteration = 0
        self.config_algorithm = config_algorithm

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
    
    #evaluate policy when step % eval_freq == 0, return rewards and lengths list of n_eval_episodes elements
    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
            
            #Write train stats at the end of every training iteration
            #on_train_result(self.training_env, self.training_iteration, self.file_name, self.config_algorithm)
            self.training_iteration +=1
            #evaluation results
            write_eval_stats(episode_rewards, episode_lengths, self.file_name)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        return True

#-------------- Custom Classes ------------#

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
    # parser.add_argument('-t', '--tune-hps', dest='tune_hps', action='store', default=False, type=bool,
    #                    help='Used for tuning the hyperparameters that can be used for experiments later.'
    #                    ' A Cartesian product of different configuration values for the experiment will be taken and ordered as a list and this number corresponds to the configuration number in this list.'
    #                    ' Please look in to the code for details.')
    args = parser.parse_args()
    print("Parsed args:", args)
    return args

# Callback to write training stats to CSV file at end of every training iteration
def on_train_result(train_env, training_iteration, filename, config_algorithm):
    # Writes every iteration, would slow things down. #hack
    fout = open(filename, 'a') #hardcoded
    fout.write(str(training_iteration) + ' ' + config_algorithm + ' ')
    for config_type, config_dict in var_configs_deepcopy.items():
        for key in config_dict:
            if config_type == "env":
                if key == 'reward_noise':
                    fout.write(str(info["result"]["config"]["env_config"]['reward_noise_std']) + ' ') #hack
                elif key == 'transition_noise' and info["result"]["config"]["env_config"]["state_space_type"] == "continuous":
                    fout.write(str(info["result"]["config"]["env_config"]['transition_noise_std']) + ' ') #hack
                else:
                    fout.write(str(info["result"]["config"]["env_config"][key]).replace(' ', '') + ' ')
            elif config_type == "agent":
                if config_algorithm == 'SAC' and key == "critic_learning_rate":
                    real_key = "lr" #hack due to Ray's weird ConfigSpaces
                    fout.write(str(info["result"]["config"]['optimization'][key]).replace(' ', '') + ' ')
                elif config_algorithm == 'SAC' and key == "fcnet_hiddens":
                    #hack due to Ray's weird ConfigSpaces
                    fout.write(str(info["result"]["config"]["Q_model"][key]).replace(' ', '') + ' ')
                else:
                    fout.write(str(info["result"]["config"][key]).replace(' ', '') + ' ')
            elif config_type == "model":
                # if key == 'conv_filters':
                fout.write(str(info["result"]["config"]["model"][key]).replace(' ', '') + ' ')

    # Write train stats
    timesteps_total = info["result"]["timesteps_total"] # also has episodes_total and training_iteration
    episode_reward_mean = info["result"]["episode_reward_mean"] # also has max and min
    # print("Custom_metrics: ", info["result"]["step_reward_mean"], info["result"]["step_reward_max"], info["result"]["step_reward_min"])
    episode_len_mean = info["result"]["episode_len_mean"]

    fout.write(str(timesteps_total) + ' ' + str(episode_reward_mean) +
               ' ' + str(episode_len_mean) + '\n') # timesteps_total always HAS to be the 1st written: analysis.py depends on it
    fout.close()

    # print("##### hack_filename: ", hack_filename)
    # print(os.getcwd())

    # We did not manage to find an easy way to log evaluation stats for Ray without the following hack which demarcates the end of a training iteration in the evaluation stats file
    hack_filename_eval = stats_file_name + '_eval.csv'
    fout = open(hack_filename_eval, 'a') #hardcoded
    fout.write('#HACK STRING EVAL' + "\n")
    fout.close()

    info["result"]["callback_ok"] = True

# Used in callback after every episode has ended during evaluation
# replaces: def on_episode_end(info)
def write_eval_stats(episode_rewards, episode_lengths, stats_file_name = ""):
    eval_filename = stats_file_name + '_eval.csv'
    fout = open(eval_filename, 'a') #hardcoded
    for reward_this_episode, length_this_episode in zip(episode_rewards, episode_lengths):
        fout.write(str(reward_this_episode) + ' ' + str(length_this_episode) + "\n")
    fout.close()

def deepmerge(a, b, path=None):
    '''Merges dict b into dict a

    Based on: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries/7205107#7205107
    '''
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def cartesian_prod(args, config):
    value_tuples = []
    for config_type, config_dict in config.var_configs.items():
        for key in config_dict:
            assert type(config.var_configs[config_type][key]) == list, "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
            value_tuples.append(config.var_configs[config_type][key])
            
    cartesian_product_configs = list(itertools.product(*value_tuples))
    print("Total number of configs. to run:", len(cartesian_product_configs))

    if args.config_num is None:
        pass
    else:
        cartesian_product_configs = [cartesian_product_configs[args.config_num]]
    
    return cartesian_product_configs

def process_dictionaries(config, current_config, var_env_configs, var_agent_configs, var_model_configs):
    algorithm = config.algorithm
    agent_config = config.agent_config
    model_config = config.model_config
    env_config = config.env_config

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

    #hacks begin:
    if model_config["model"]["use_lstm"]:
        model_config["model"]["max_seq_len"] = env_config["env_config"]["delay"] + env_config["env_config"]["sequence_length"] + 1

    if algorithm == 'DDPG': ###TODO Find a better way to enforce these??
        agent_config["actor_lr"] = agent_config["critic_lr"]
        agent_config["actor_hiddens"] = agent_config["critic_hiddens"]
    elif algorithm == 'TD3':
        agent_config["target_noise_clip"] = agent_config["target_noise_clip"] * agent_config["target_noise"]

    # else: #if algorithm == 'SAC':
    if env_config["env_config"]["state_space_type"] == 'continuous':
        env_config["env_config"]["action_space_dim"] = env_config["env_config"]["state_space_dim"]
    
    return agent_config, model_config, env_config, algorithm

def act_fnc_from_name(key):
    if(key == "tanh"):
        return nn.Tanh
    elif(key == "sigmoid"):
        return nn.Sigmoid
    elif(key == "leakyrelu"):
        return nn.LeakyReLU
    else: #default
        return nn.ReLU

def process_agent_dict(algorithm, agent_config, model_config, env):
    policy_kwargs = {}
    if algorithm == 'DQN':
        #change keys in dictionaries to something baselines understands
        valid_keys =  ['buffer_size', "learning_starts", "tau","gamma", "train_freq", "gradient_steps", "n_episodes_rollout", "target_update_interval","exploration_fraction","exploration_initial_eps","exploration_final_eps","max_grad_norm","tensorboard_log","learning_rate","batch_size","target_update_interval","lr_schedule"]

        exchange_keys = [
                        ('lr','learning_rate'),
                        ('train_batch_size', 'batch_size'),
                        ('target_network_update_freq','target_update_interval')]
        for key_tuple in exchange_keys:
            old_key, new_key = key_tuple
            if(old_key in agent_config):
                agent_config[new_key] = agent_config.pop(old_key)
        #remove keys from dictionary which are not configurable by baselines
        for key in list(agent_config.keys()):
            if( key not in valid_keys ):
                agent_config.pop(key)

    #-------------- Model configuration ------------#
    #decide whether to use cnn or mlp, taken from ray code..
    # Discrete/1D obs-spaces.
    if isinstance(env.observation_space, gym.spaces.Discrete) or \
            len(env.observation_space.shape) <= 2:
        feat_ext ="MlpPolicy"
    else:# Default Conv2D net.
        feat_ext = "CnnPolicy"   
    agent_config['policy'] = feat_ext
    cnn_config = {} 

    if(feat_ext == "MlpPolicy"): #mlp config
        if("fcnet_activation" in model_config['model']):
            policy_kwargs['activation_fn'] = act_fnc_from_name( model_config['model']["fcnet_activation"])
        if("fcnet_hiddens" in model_config['model']):
            policy_kwargs['net_arch'] = model_config['model']["fcnet_hiddens"]
    #this is to create a custom CNN since there's no implementation currently
    else: #cnn config
        policy_kwargs['net_arch'] = model_config['model']["fcnet_hiddens"]
        if("conv_activation" in model_config['model'] ):
            cnn_config["activation_fn"] = act_fnc_from_name( model_config['model']["conv_activation"] )
        if("conv_filters" in model_config['model'] ):
            cnn_config["cnn_layers"] = model_config['model']["conv_filters"]
    
    #custom
    if( feat_ext == "CnnPolicy"):
        policy_kwargs['features_extractor_class'] = VisionNet
        policy_kwargs["features_extractor_kwargs"] = cnn_config

    #add policy arguments to agent configuration
    if('lr_schedule'in agent_config.keys()): #schedule is part of model instead of agent in baselines
        policy_kwargs['lr_schedule'] = agent_config['lr_schedule']
    agent_config['policy_kwargs'] = policy_kwargs

    return agent_config

def main(args):
    #print("Config file", os.path.abspath(args.config_file)) # 'experiments/dqn_seq_del.py'
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

    #------------------------- Variable configurations  ----------------------------#
    var_configs_deepcopy = copy.deepcopy(config.var_configs)

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

    print('# Algorithm, state_space_size, action_space_size, delay, sequence_length, reward_density, make_denser, terminal_state_density, transition_noise, reward_noise ')

    configs_to_print = ''
    for config_type, config_dict in var_configs_deepcopy.items():
        if config_type == 'env':
            for key in config_dict:
                configs_to_print += str(config_dict[key]) + ', '

    print(config.algorithm, configs_to_print)

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
    cartesian_product_configs = cartesian_prod(args, config)

    #------------ Run every configuration  ------------------ #
    pp = pprint.PrettyPrinter(indent=4)
    for current_config in cartesian_product_configs:
        #------------ Dictionaries setup  ------------------ #
        agent_config, model_config, env_config, algorithm = process_dictionaries(config, current_config, var_env_configs, var_agent_configs, var_model_configs) #modify to match baselines syntax

        #------------ Env init  ------------------ #
        env = gym.make(env_config['env'],**env_config['env_config'])
        env = Monitor(env, None)

        #Evaluation environment
        #limit max timesteps in evaluation = horizon
        eval_horizon = 100
        eval_env = gym.make(env_config['env'],**env_config['env_config']) 
        eval_env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = eval_horizon
        eval_env = TimeLimit(eval_env, max_episode_steps = eval_horizon) #horizon
        print("Env config:",)
        pp.pprint(env_config)
        
        #------------ Agent init  ------------------ #
        # Create the agent
        timesteps_per_iteration = agent_config['timesteps_per_iteration']
        agent_config_p = process_agent_dict(algorithm, agent_config.copy(), model_config, env)
        if config.algorithm == 'DQN':
            model = DQN(env = env, **agent_config_p, verbose=1)
        elif config.algorithm == 'DDPG': #hack
            model = DDPG(env = env, **agent_config_p)
        else: #'SAC'
            model = SAC(env = env, **agent_config_p)

        #------------ evaluation / training ------------------ #
        if env_config["env"] in ["HopperWrapper-v3", "HalfCheetahWrapper-v3"]: #hack
            timesteps_total = 500000
        else:
            if algorithm == 'DQN':
                timesteps_total = 20000
            elif algorithm == 'A3C': #hack
                timesteps_total = 150000
            else: #if algorithm == 'DDPG': #hack
                timesteps_total = 20000

        #Define evaluation
        evaluation_interval = 1 #every x training iterations
        eval_frequency = timesteps_per_iteration * evaluation_interval
        eval_callback = CustomCallback(eval_env, n_eval_episodes=10, eval_freq=eval_frequency, \
                        deterministic=True, file_name = stats_file_name, config_algorithm = algorithm)
        #Train
        model.learn(callback= eval_callback, total_timesteps = timesteps_total)
        rewards = env.get_episode_rewards()
        espisode_lengths = env.get_episode_lengths()
        episode_times = env.get_episode_times()
        total_steps = env.get_total_steps()
        # Evaluate the agent at the end of training?
        #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    end = time.time()
    print("No. of seconds to run:", end - start)


if __name__ == '__main__':
    args = parse_args()
    main(args)