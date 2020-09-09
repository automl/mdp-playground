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

import tensorflow as tf
import stable_baselines as sb
from stable_baselines import DQN, DDPG, SAC, A2C, TD3
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from typing import Any, Dict, List, Optional, Union

from functools import reduce
import time
import itertools
import pprint

class CustomCallback(sb.common.callbacks.BaseCallback):
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
        config_algorithm: str ="",
        var_configs = None
    ):
        super(CustomCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.file_name = file_name
        self.training_iteration = 0
        self.timesteps_per_iteration = 0
        self.last_iter_ts = 0
        self.config_algorithm = config_algorithm
        self.var_configs = var_configs

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
    
    #evaluate policy when step % eval_freq == 0, return rewards and lengths list of n_eval_episodes elements
    def _on_step(self):
        #count steps per iteration
        self.timesteps_per_iteration+=1
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = sb.common.evaluation.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
            
            #write csv results
            self.write_train_result()
            self.write_eval_results(episode_rewards, episode_lengths)
            # self.last_iter_ts = self.timesteps_per_iteration
            # self.timesteps_per_iteration = 0
            self.training_iteration+=1

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        return True

    # Used in callback after every episode has ended during evaluation
    # replaces: def on_episode_end(info)
    def write_eval_results(self, episode_rewards, episode_lengths):
        eval_filename = self.file_name + '_eval.csv'
        fout = open(eval_filename, 'a') #hardcoded
        for reward_this_episode, length_this_episode in zip(episode_rewards, episode_lengths):
            fout.write(str(reward_this_episode[0]) + ' ' + str(length_this_episode) + "\n")
        fout.close()
            
    # Write training stats to CSV file at end of every training iteration
    def write_train_result(self):
        env, training_iteration, timesteps_total = self.training_env, self.training_iteration, self.timesteps_per_iteration
        file_name, config_algorithm, var_configs = self.file_name, self.config_algorithm, self.var_configs
        # Writes every iteration, would slow things down. #hack
        fout = open(self.file_name + '.csv', 'a') #hardcoded
        fout.write(str(training_iteration) + ' ' + config_algorithm + ' ')
        for config_type, config_dict in var_configs.items():
            for key in config_dict:
                if config_type == "env":
                    env_config = env.config
                    if key == 'reward_noise':
                        fout.write(str(env_config['reward_noise_std']) + ' ') #hack
                    elif key == 'transition_noise' and env_config["state_space_type"] == "continuous":
                        fout.write(str(env_config['transition_noise_std']) + ' ') #hack
                    else:
                        fout.write(str(env_config[key]).replace(' ', '') + ' ')
                elif config_type == "agent":
                    fout.write(str(getattr(self.model, key)).replace(' ', '') + ' ')
                elif config_type == "model":
                    if(key == "net_arch"):#this is kwargs as it is sent to visionet
                        fout.write(str(getattr(self.model, "policy_kwargs")["kwargs"][key]).replace(' ', '') + ' ')
                    else:
                        fout.write(str(getattr(self.model, "policy_kwargs")[key]).replace(' ', '') + ' ')

        # Write train stats
        rewards = env.get_episode_rewards()
        episode_lengths = env.get_episode_lengths()
        episode_times = env.get_episode_times()
        total_steps = env.get_total_steps()
        if(len(rewards) == 0):#no episodes yet
            episode_reward_mean = np.mean( env.rewards )
            episode_len_mean = env.total_steps
        else:
            #episode stats are from all steps taken in env then we need to count "iterations"
            episode_reward_mean = np.mean(rewards)
            episode_len_mean =  np.mean(episode_lengths)

        fout.write(str(timesteps_total) + ' ' + str(episode_reward_mean) +
                ' ' + str(episode_len_mean) + '\n') # timesteps_total always HAS to be the 1st written: analysis.py depends on it
        fout.close()

        # We did not manage to find an easy way to log evaluation stats for Ray without the following hack which demarcates the end of a training iteration in the evaluation stats file
        hack_filename_eval = self.file_name + '_eval.csv'
        fout = open(hack_filename_eval, 'a') #hardcoded
        fout.write('#HACK STRING EVAL' + "\n")
        fout.close()

        #info["result"]["callback_ok"] = True
        return True

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
    if "use_lstm" in model_config["model"]:
        if model_config["model"]["use_lstm"]:#if true
            model_config["model"]["max_seq_len"] = env_config["env_config"]["delay"] + env_config["env_config"]["sequence_length"] + 1

    elif algorithm == 'TD3':
        if("target_policy_noise" in agent_config):
            agent_config["target_noise_clip"] = agent_config["target_noise_clip"] * agent_config["target_policy_noise"]

    # else: #if algorithm == 'SAC':
    if("state_space_type" in env_config["env_config"]):
        if env_config["env_config"]["state_space_type"] == 'continuous':
            env_config["env_config"]["action_space_dim"] = env_config["env_config"]["state_space_dim"]
    
    return agent_config, model_config, env_config

def vision_net(scaled_images, kwargs):
    if ("act_fun" in kwargs):
        activ_fn = kwargs['act_fun']
    else: #default
        activ_fn = tf.nn.relu

    output, i = scaled_images, 0
    for i, layer_def in enumerate(kwargs['net_arch'][:-1], start=1):
        n_filters, kernel, stride = layer_def
        scope = "c%d"%(i)
        output = activ_fn(conv(output, scope, n_filters = n_filters, filter_size = kernel[0], stride = stride, pad="SAME", init_scale=np.sqrt(2)))
    
    #last layer has valid padding
    n_filters, kernel, stride = kwargs['net_arch'][-1]
    output = activ_fn(conv(output, scope = "c%d"%(i+1), n_filters = n_filters, filter_size = kernel[0], stride = stride, pad="VALID", init_scale=np.sqrt(2)))
    output = conv_to_fc(output) #No linear layer
    #output = activ_fn(linear(output, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    return output

def act_fnc_from_name(key):
    if(key == "tanh"):
        return tf.nn.tanh
    elif(key == "sigmoid"):
        return tf.nn.sigmoid
    elif(key == "leakyrelu"):
        return tf.nn.leaky_relu
    else: #default
        return tf.nn.relu

#change config.agent_config, config.var_model_config and config.var_agent_config to baselines framework
def agent_to_baselines(config):
    policy_kwargs = {}
    algorithm, agent_config, model_config = config.algorithm, config.agent_config, config.model_config
    try:
        var_agent_configs = config.var_agent_configs
    except AttributeError:
        var_agent_configs = {} #var_model_configs does not exist
    try:
        var_model_configs = config.var_model_configs
    except AttributeError:
        var_model_configs = {} #var_model_configs does not exist
    
    valid_keys = ["gamma", "buffer_size", "batch_size", "learning_starts","timesteps_per_iteration"]#common
    agent_to_model = []#none
    #Check correct keys for each algorithm
    if algorithm == 'DQN':
        #change keys in dictionaries to something baselines understands
        valid_keys += ["exploration_fraction","exploration_final_eps", "exploration_initial_eps",
                    "train_freq", "double_q",
                    "prioritized_replay", "prioritized_replay_alpha", 
                    "param_noise", "learning_rate", "target_network_update_freq"]

        exchange_keys = [('noisy', 'param_noise'),
                        ('lr','learning_rate'),
                        ('train_batch_size', 'batch_size')]
        #Move dueling to policy parameters
        agent_to_model+=[("dueling","dueling")]
    elif algorithm == "DDPG":
        #memory_policy can be used to select PER
        #random_exploration, param_noise, action_noise parameters aditional
        valid_keys += ["critic_lr", "actor_lr", "tau","critic_l2_reg",\
                        "clip_norm", "nb_rollout_steps", "nb_train_steps"]
        valid_keys.remove("learning_starts") #For some reason it's the only one that does not have this implemented
        exchange_keys = [
                ("l2_reg", "critic_l2_reg"),
                ("grad_norm_clipping", "clip_norm"),
                ('train_batch_size',   'batch_size')]
        # Because of how DDPG is implemented it will perform 100 rollout steps and then 50 train steps
        # This needs to be changed s.t. it does one rollout step and one training step
        agent_config["nb_rollout_steps"] = 1
        agent_config["nb_train_steps"] = 1
        agent_to_model+=[("actor_hiddens", "layers"),("critic_hiddens", "layers")]#cannot specify different nets
        #SPECIAL CASE, learning rates SHOULD NOT be none :c
        for key in ["critic_lr", "actor_lr"]:
            key_not_none = agent_config.get(key) 
            if (key in agent_config):
                if(not key_not_none): #key==none
                    agent_config.pop(key) #remove to get default value

    elif algorithm == "TD3":
        #memory_policy can be used to select PER; policy is always smoothened
        #random_exploration, param_noise, action_noise parameters aditional
        valid_keys += ["learning_rate", "policy_delay","tau","train_freq","gradient_steps",\
                       "target_policy_noise","target_noise_clip","action_noise"]
        exchange_keys = [
                ("target_noise", "target_policy_noise"),
                ("critic_lr", "learning_rate"),
                ("actor_lr", "learning_rate"),
                ('train_batch_size',   'batch_size')]
        agent_to_model+=[("actor_hiddens", "layers"),("critic_hiddens", "layers")]
        #SPECIAL CASE, learning rates SHOULD NOT be none :c
        for key in ["critic_lr, actor_lr"]:
            key_not_none = agent_config.get(key) 
            if (key in agent_config):
                if(not key_not_none):
                    agent_config.pop(key) #remove to get default value
        # Because of how TD3 is implemented it will perform 100 grad steps every 100 steps
        # This needs to be changed s.t. it does one rollout step and one training step
        agent_config["train_freq"] = 1
        agent_config["gradient_steps"] = 1
        agent_config["action_noise"] = NormalActionNoise(mean=0, sigma=0) #s.t. it clips actions between [-1,1]
    elif algorithm == "SAC":
        valid_keys += ["learning_rate", "tau", "ent_coef",
                       "target_update_interval","clip_norm","target_entropy"]
        exchange_keys = [
                ("entropy_learning_rate","learning_rate"), #same learning rate for 
                ("critic_learning_rate", "learning_rate"), #entropy 
                ("actor_learning_rate", "learning_rate"), #actor and critic
                ("target_network_update_freq","target_update_interval"),
                ("grad_norm_clipping", "clip_norm"),
                ('train_batch_size',   'batch_size')]
        agent_to_model+=[("fcnet_hiddens", "layers")]
        #-------- Add init entropy coef ------------#
        if("initial_alpha" in agent_config.keys()):
            agent_config["ent_coef"] = "auto_%.3f"%agent_config["initial_alpha"] #idk why but that's how baselines initializes this thing..
            agent_config.pop("initial_alpha")
        #var agent configs initial_alpha
        #idk why but that's how baselines initializes this thing..
        if("agent" in config.var_configs):
            if("initial_alpha" in var_agent_configs):
                var_agent_configs["ent_coef"] = ["auto_%.3f"%coef for coef in var_agent_configs["initial_alpha"]]
                var_agent_configs.pop("initial_alpha")
        # ----------- special case ---------#
        #should be at least one
        if("target_network_update_freq" in config.agent_config):
            key_val = agent_config["target_network_update_freq"]
            agent_config["target_network_update_freq"] = 1 if key_val <= 0 else key_val 
        #-------- Take things out of optimization  ------------#
        #Special case: Optimization
        if("optimization" in agent_config):
            for k in config.agent_config["optimization"].keys():
                key_value = config.agent_config["optimization"][k]
                if(key_value): #not none
                    agent_config[k] = key_value
        #-------- Take things out of Q-model and policy_model  ------------#
        #Can only specify one model for everything
        for move_key in ["Q_model", "policy_model"]:
            if(move_key in model_config):
                if("model" not in model_config):#init
                    model_config["model"]={}
                for k in config.model_config[move_key].keys():
                    model_config["model"][k] =  config.model_config[move_key][k]
            model_config.pop(move_key)#Remove from model_config

    #-------- Change keys from Agent to Model dict when needed--------#
    for key_tuple in agent_to_model:
        old_key, new_key = key_tuple
        #-------- Agent config --------#
        key_exists = agent_config.get(old_key) 
        if key_exists:#If key exists and is not none
            policy_kwargs[new_key] =  agent_config.pop(old_key) #Move key to model config
        # #-------- Var agent config --------#
        key_exists = var_agent_configs.get(old_key) 
        if key_exists:#If key exists and is not none
            var_model_configs[new_key] =  var_agent_configs.pop(old_key) #Move key to model config

    if('fcnet_hiddens' in agent_config.keys()):
        policy_kwargs['layers'] = agent_config['fcnet_hiddens']
        agent_config.pop('fcnet_hiddens')

    #-------- Agent config --------#
    #change agent keys
    for key_tuple in exchange_keys:
        old_key, new_key = key_tuple
        #-------- Agent config --------#
        if(old_key in agent_config):
            agent_config[new_key] = agent_config.pop(old_key)
        #-------- Var agent config --------#
        if(old_key in var_agent_configs):
            var_agent_configs[new_key] =  var_agent_configs.pop(old_key)

    #remove keys from dictionary which are not configurable by baselines
    for key in list(agent_config.keys()):
        if( key not in valid_keys ):#delete invalid keys
            agent_config.pop(key)

    #-------- Var agent config --------#
    #remove keys from dictionary which are not configurable by baselines
    for key in list(var_agent_configs.keys()):
        if( key not in valid_keys ):#delete invalid keys
            var_agent_configs.pop(key) 

    #----------- Model config ----------------#
    valid_keys = ["act_fun", "net_arch", "feature_extractor", "layers", "use_lstm"]
    exchange_keys = [("fcnet_activation", "act_fun"),
                     ("conv_activation", "act_fun"),
                     ("actor_hidden_activation", "act_fun"),
                     ("actor_hidden_activation", "act_fun"),
                     ("fcnet_hiddens", "layers"),
                     ("actor_hiddens", "layers"),
                     ("critic_hiddens","layers"),
                     ("conv_filters", "net_arch")]

    #change keys
    for key_tuple in exchange_keys:
        old_key, new_key = key_tuple
        #-------- model config --------#
        if(old_key in model_config["model"]):
            model_config["model"][new_key] = model_config["model"].pop(old_key)

        #-------- Var model config --------#
        if(old_key in var_model_configs):
            var_model_configs[new_key] = var_model_configs.pop(old_key)
    
    #remove invalid keys
    for key in list(model_config["model"].keys()):
        if( key not in valid_keys ):
            model_config["model"].pop(key)

    #-------- Var model config - delete invalid keys -------#
    for key in list(var_model_configs.keys()):
        if( key not in valid_keys ):
            var_model_configs.pop(key)
    
    #-------- set model config -------#
    agent_config['policy_kwargs'] = policy_kwargs
    config.agent_config = agent_config
    config.model_config = model_config
    if bool(var_agent_configs): #variable configs
        config.var_configs["agent"] = var_agent_configs
    if bool(var_model_configs):
        config.var_configs["model"] = var_model_configs
    return config, var_agent_configs, var_model_configs

#change config.model_config to baselines framework/ decide MLP or CNN policy
def model_to_policy_kwargs(env, config):
    model_config = config.model_config
    #-------------- Model configuration ------------#
    #decide whether to use cnn or mlp, taken from ray code..
    # Discrete/1D obs-spaces.
    if isinstance(env.observation_space, gym.spaces.Discrete) or \
            len(env.observation_space.shape) <= 2:
        feat_ext ="mlp"
    else:# Default Conv2D net.
        feat_ext = "cnn"

    #Move model config(already on baselines framework) to policy_kwargs
    policy_kwargs, cnn_config = {}, {}
    for key in model_config["model"].keys():
        if( key == "act_fun" ):
            policy_kwargs[key] = act_fnc_from_name( model_config["model"][key] )
        else:
            policy_kwargs[key] = model_config["model"][key]
    
    if("use_lstm" in policy_kwargs.keys()):
        #use_lstm does not exist in baselines, here is used to define the policy
        use_lstm =  policy_kwargs.pop("use_lstm")

    #Custom Feature extractor
    if(feat_ext == "cnn" and ("net_arch" in policy_kwargs)):
        cnn_config["act_fun"] = act_fnc_from_name( policy_kwargs.pop("act_fun") )
        cnn_config["net_arch"] =  policy_kwargs.pop("net_arch")
        #Uncomment to pass through custom cnn
        policy_kwargs['cnn_extractor'] = vision_net
        policy_kwargs["kwargs"] = cnn_config

    policy_kwargs['feature_extraction'] = feat_ext 
    # #add policy arguments to agent configuration
    # if('lr_schedule'in agent_config.keys()): #schedule is part of model instead of agent in baselines
    #     agent_config['policy_kwargs']['lr_schedule'] = agent_config['lr_schedule']
    
    return policy_kwargs

def main():   
    #-------------------------  init configuration ----------------------------#
    #print("Config file", os.path.abspath(args.config_file)) # 'experiments/dqn_seq_del.py'
    args = parse_args()
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
    config, var_agent_configs, var_model_configs = agent_to_baselines(config)
    var_configs_deepcopy = copy.deepcopy(config.var_configs) #hack because this needs to be read in on_train_result and trying to read config there raises an error because it's been imported from a Python module and I think they try to reload it there.

    if "env" in config.var_configs:
        var_env_configs = config.var_configs["env"] #hack
    else:
        var_env_configs = []
    # Modified in agent_to_baselines    
    # if "agent" in config.var_configs:
    #     var_agent_configs = config.var_configs["agent"] #hack
    # else:
    #     var_agent_configs = []
    # if "model" in config.var_configs:
    #     var_model_configs = config.var_configs["model"] #hack
    # else:
    #     var_model_configs = []
    config_algorithm = config.algorithm #hack, used on on_train_result

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
    for i, current_config in enumerate(cartesian_product_configs):
        print("Configuration: %d \t %d remaining"%(i, len(cartesian_product_configs) - i ))
        #------------ Dictionaries setup  ------------------ #
        agent_config, model_config, env_config = process_dictionaries(config, current_config, var_env_configs, var_agent_configs, var_model_configs) #hacks

        #------------ Env init  ------------------ #
        #env = RLToyEnv(env_config["env_config"]) #alternative
        if( "horizon" in env_config ):
            train_horizon = env_config["horizon"]
        else:
            train_horizon = None
        env = gym.make(env_config['env'],**env_config['env_config'])
        env = TimeLimit(env, max_episode_steps = train_horizon) #horizon
        env = sb.bench.monitor.Monitor(env, None)
        #check_env(env)
        #limit max timesteps in training too... (idk why but seems like ray does this (?))
        
        #limit max timesteps in evaluation
        eval_horizon = 100
        eval_env = gym.make(env_config['env'],**env_config['env_config'])
        eval_env = TimeLimit(eval_env, max_episode_steps = eval_horizon) #horizon
        print("Env config:",)
        pp.pprint(env_config)

        #------------ Agent init  ------------------ #
        # Create the agent
        if("timesteps_per_iteration" in agent_config):
            timesteps_per_iteration = agent_config["timesteps_per_iteration"]
            agent_config_baselines = copy.deepcopy(agent_config)
            agent_config_baselines.pop('timesteps_per_iteration') #this is not part of baselines parameters, need to separate
        else:
            timesteps_per_iteration = 1000#default

        #Set model parameters
        policy_kwargs = model_to_policy_kwargs(env, config) #return policy_kwargs
        agent_config_baselines["policy_kwargs"] = policy_kwargs
        
        #Use feed forward policies and specify cnn feature extractor in configuration
        if config.algorithm == 'DQN':
            model = DQN(env = env, policy = sb.deepq.policies.FeedForwardPolicy, **agent_config_baselines, verbose=1)
        elif config.algorithm == 'DDPG': #hack
            model = DDPG(env = env, policy = sb.ddpg.policies.FeedForwardPolicy, **agent_config_baselines, verbose=1 )
        elif config.algorithm == "TD3":
            model = TD3(env = env, policy = sb.td3.policies.FeedForwardPolicy, **agent_config_baselines, verbose=1 )
        else: #'SAC
            model = SAC(env = env, policy = sb.sac.policies.FeedForwardPolicy ,**agent_config_baselines, verbose=1)

        #------------ train/evaluation ------------------ #
        if env_config["env"] in ["HopperWrapper-v3", "HalfCheetahWrapper-v3"]:
            timesteps_total = 500000
        else:
            timesteps_total = 20000 #DQN, DDPG, TD3, SAC

        # train your model for n_iter timesteps
        #Define evaluation
        evaluation_interval = 1 #every x training iterations
        eval_frequency = timesteps_per_iteration * evaluation_interval
        eval_callback = CustomCallback(eval_env, n_eval_episodes=10, eval_freq=eval_frequency, \
                        deterministic=True, file_name = stats_file_name, \
                        config_algorithm = config.algorithm, var_configs = var_configs_deepcopy)
        #Train
        learn_params = {"callback": eval_callback,
                        "total_timesteps": timesteps_total}
        if(config.algorithm == "DDPG"): #Log interval is handled differently for each algorithm, e.g. each log_interval episodes(DQN) or log_interval steps(DDPG).
            learn_params["log_interval"] = timesteps_per_iteration
        else:
            learn_params["log_interval"] = timesteps_per_iteration//10
        
        model.learn(**learn_params)

    end = time.time()
    print("No. of seconds to run:", end - start)


if __name__ == '__main__':
    main()