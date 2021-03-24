'''Script to run experiments on MDP Playground.
Takes a configuration file, experiment name and config number to run as optional arguments.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import warnings
import logging

import ray
from ray import tune
from ray.rllib.utils.seed import seed as rllib_seed
import mdp_playground
from mdp_playground.envs import RLToyEnv
from ray.tune.registry import register_env
import sys, os
import argparse

register_env("RLToy-v0", lambda config: RLToyEnv(**config))
register_env("GymEnvWrapper-Atari", lambda config: create_gym_env_wrapper_atari(config))
register_env("GymEnvWrapperFrameStack-Atari", lambda config: create_gym_env_wrapper_frame_stack_atari(config))
register_env("RLToy-v0", lambda config: RLToyEnv(**config))

def create_gym_env_wrapper_atari(config):
    from gym.envs.atari import AtariEnv
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    ae = AtariEnv(**config["AtariEnv"])
    gew = GymEnvWrapper(ae, **config) ##IMP Had initially thought to put this config in config["GymEnvWrapper"] but because of code below which converts var_env_configs to env_config, it's best to leave those configs as top level configs in the dict!
    return gew



def create_gym_env_wrapper_frame_stack_atari(config): #hack ###TODO remove?
    '''When using frameStack GymEnvWrapper should wrap AtariEnv using wrap_deepmind_ray and therefore this function sets "wrap_deepmind_ray": True and 'frame_skip': 1 inside config so as to keep config same as for create_gym_env_wrapper_atari above and reduce manual errors when switching between the 2.
    '''
    config["wrap_deepmind_ray"] = True #hack
    config["frame_skip"] = 1 #hack
    from gym.envs.atari import AtariEnv
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    import gym
    game = config["AtariEnv"]["game"]
    game = ''.join([g.capitalize() for g in game.split('_')])
    ae = gym.make('{}NoFrameskip-v4'.format(game))
    gew = GymEnvWrapper(ae, **config) ##IMP Had initially thought to put this config in config["GymEnvWrapper"] but because of code below which converts var_env_configs to env_config, it's best to leave those configs as top level configs in the dict!
    return gew



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__) # docstring at beginning of the file is stored in __doc__
    parser.add_argument('-c', '--config-file', dest='config_file', action='store', default='default_config',
                    help='Configuration file containing configuration to run experiments. It must be a Python file so config can be given programmatically. There are 2 types of configs - VARIABLE CONFIG across the experiments and STATIC CONFIG across the experiments. \nVARIABLE CONFIGS: The OrderedDicts var_env_configs, var_agent_configs and var_model_configs hold configuration options that are variable for the environment, agent and model across the current experiment. For each configuration option, the option is the key in the dict and its value is a list of values it can take for the current experiment. A Cartesian product of these lists is taken to generate various possible configurations to be run. For example, you might want to vary "delay" for the current experiment. Then "delay" would be a key in var_env_configs dict and its value would be a list of values it can take. Because Ray does not have a common way to address this specification of configurations for its agents, there are a few hacky ways to set var_agent_configs and var_model_configs currently. Please see sample experiment config files in the experiments directory to see how to set the values for a given algorithm. \nSTATIC CONFIGS: env_config, agent_config and model_config are dicts which hold the static configuration for the current experiment as a normal Python dict.') ####TODO Update docs regarding how to get configs to run: i.e., Cartesian product, or random, etc.
    parser.add_argument('-e', '--exp-name', dest='exp_name', action='store', default='mdpp_default_experiment',
                    help='The user-chosen name of the experiment. This is used as the prefix of the output files (the prefix also contains config_num if that is provided). It will save stats to 2 CSV files, with the filenames as the one given as argument'
                    ' and another file with an extra "_eval" in the filename that contains evaluation stats during the training. Appends to existing files or creates new ones if they don\'t exist.')
    parser.add_argument('-n', '--config-num', dest='config_num', action='store', default=None, type=int,
                    help='Used for running the configurations of experiments in parallel. This is appended to the prefix of the output files (after exp_name).'
                    ' A Cartesian product of different configuration values for the experiment will be taken and ordered as a list and this number corresponds to the configuration number in this list.'
                    ' Please look in to the code for details.')
    parser.add_argument('-a', '--agent-config-num', dest='agent_config_num', action='store', default=None, type=int,
                    help='Used for running the configurations of experiments in parallel. This is appended to the prefix of the output files (after exp_name).') ###TODO Remove? #hack to run 1000 x 1000 env configs x agent configs. Storing all million of them in memory may be too inefficient?
    parser.add_argument('-m', '--save-model', dest='save_model', action='store', default=False, type=bool,
                    help='Option to save trained NN model at the end of training.')
    parser.add_argument('-t', '--framework-dir', dest='framework_dir', action='store', default='/tmp/', type=str,
                    help='Prefix of directory to be used by underlying framework (e.g. Ray Rllib, Stable Baselines 3). This name will be passed to the framework.')
    # parser.add_argument('-t', '--tune-hps', dest='tune_hps', action='store', default=False, type=bool,
    #                    help='Used for tuning the hyperparameters that can be used for experiments later.'
    #                    ' A Cartesian product of different configuration values for the experiment will be taken and ordered as a list and this number corresponds to the configuration number in this list.'
    #                    ' Please look in to the code for details.')

    args = parser.parse_args()
    print("Parsed args:", args)
    return args

def main(args):

    if args.config_file[-3:] == '.py':
        args.config_file = args.config_file[:-3]

    config_file_path = os.path.abspath('/'.join(args.config_file.split('/')[:-1]))
    # print("config_file_path:", config_file_path)
    sys.path.insert(1, config_file_path) #hack
    import importlib
    config = importlib.import_module(args.config_file.split('/')[-1], package=None)
    print("Number of seeds for environment:", config.num_seeds)
    print("Configuration number(s) that will be run:", "all" if args.config_num is None else args.config_num)


    # import default_config
    # print("default_config:", default_config)
    # print(os.path.abspath(args.config_file)) # 'experiments/dqn_seq_del.py'

    args.exp_name = os.path.abspath(args.exp_name)
    stats_file_name = args.exp_name

    if args.config_num is not None:
        stats_file_name += '_' + str(args.config_num)
    # elif args.agent_config_num is not None: ###TODO Remove? If we append both these nums then, that can lead to 1M small files for 1000x1000 configs which doesn't play well with our Nemo cluster.
    #     stats_file_name += '_' + str(args.agent_config_num)

    print("Stats file being written to:", stats_file_name)


    #TODO Different seeds for Ray Trainer (TF, numpy, Python; Torch, Env), Environment (it has multiple sources of randomness too), Ray Evaluator
    from ray.rllib.models.preprocessors import OneHotPreprocessor
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_preprocessor("ohe", OneHotPreprocessor)

    log_level_ = logging.WARNING ##TODO Make a runtime argument

    if config.algorithm == 'DQN':
        ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9), temp_dir='/tmp/ray' + str(args.config_num), include_webui=False, logging_level=log_level_, local_mode=True) #webui_host='0.0.0.0'); logging_level=logging.INFO,

        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9), local_mode=True, plasma_directory='/tmp') #, memory=int(8e9), local_mode=True # local_mode (bool): If true, the code will be executed serially. This is useful for debugging. # when true on_train_result and on_episode_end operate in the same current directory as the script. A3C is crashing in local mode, so didn't use it and had to work around by giving full path + filename in stats_file_name.; also has argument driver_object_store_memory=, plasma_directory='/tmp'
    elif config.algorithm == 'A3C': #hack
        ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9), temp_dir='/tmp/ray' + str(args.config_num), include_webui=False, logging_level=log_level_)
        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9), local_mode=True, plasma_directory='/tmp')
    else:
        ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9), local_mode=True, temp_dir='/tmp/ray' + str(args.config_num), include_webui=False, logging_level=log_level_)


    var_configs_deepcopy = copy.deepcopy(config.var_configs) #hack because this needs to be read in on_train_result and trying to read config there raises an error because it's been imported from a Python module and I think they try to reload it there.
    if "timesteps_total" in dir(config):
        hacky_timesteps_total = config.timesteps_total #hack

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


    hack_filename = stats_file_name + '.csv'
    fout = open(hack_filename, 'a') #hardcoded
    fout.write('# training_iteration, algorithm, ')
    for config_type, config_dict in var_configs_deepcopy.items():
        for key in config_dict:
            # if config_type == "agent":
            #     if config_algorithm == 'SAC' and key == "critic_learning_rate":
            #         real_key = "lr" #hack due to Ray's weird ConfigSpaces
            #         fout.write(real_key + ', ')
            #     elif config_algorithm == 'SAC' and key == "fcnet_hiddens":
            #         #hack due to Ray's weird ConfigSpaces
            #         fout.write('fcnet_hiddens' + ', ')
            #     else:
            #         fout.write(key + ', ')
            # else:
            fout.write(key + ', ')
    fout.write('timesteps_total, episode_reward_mean, episode_len_mean\n')
    fout.close()


    import time
    start = time.time()

    # Ray callback to write training stats to CSV file at end of every training iteration
    def on_train_result(info):
        training_iteration = info["result"]["training_iteration"]
        # algorithm = info["trainer"]._name

        # Writes every iteration, would slow things down. #hack
        fout = open(hack_filename, 'a') #hardcoded
        fout.write(str(training_iteration) + ' ' + config_algorithm + ' ')
        for config_type, config_dict in var_configs_deepcopy.items():
            for key in config_dict:
                if config_type == "env":
                    if key == 'reward_noise':
                        fout.write('%.2e' % info["result"]["config"]["env_config"]['reward_noise_std'] + ' ') #hack
                    elif key == 'transition_noise' and info["result"]["config"]["env_config"]["state_space_type"] == "continuous":
                        fout.write('%.2e' % info["result"]["config"]["env_config"]['transition_noise_std'] + ' ') #hack
                    else:
                        field_val = info["result"]["config"]["env_config"][key]
                        if isinstance(field_val, float):
                            str_to_write = '%.2e' % field_val
                        elif type(field_val) == list:
                            str_to_write = "["
                            for elem in field_val:
                                # print(key)
                                str_to_write += '%.2e' % elem if isinstance(elem, float) else str(elem)
                                str_to_write += ","
                            str_to_write += "]"
                        else:
                            str_to_write = str(field_val).replace(' ', '')
                        str_to_write += ' '
                        fout.write(str_to_write)
                elif config_type == "agent":
                    if config_algorithm == 'SAC' and key == "critic_learning_rate":
                        real_key = "lr" #hack due to Ray's weird ConfigSpaces
                        fout.write('%.2e' % info["result"]["config"]['optimization'][key].replace(' ', '') + ' ')
                    elif config_algorithm == 'SAC' and key == "fcnet_hiddens":
                        #hack due to Ray's weird ConfigSpaces
                        str_to_write = str(info["result"]["config"]["Q_model"][key]).replace(' ', '') + ' '
                        fout.write(str_to_write)
                    # elif config_algorithm == 'SAC' and key == "policy_model":
                    #     #hack due to Ray's weird ConfigSpaces
                    #     pass
                        # fout.write(str(info["result"]["config"][key]['fcnet_hiddens']).replace(' ', '') + ' ')
                    else:
                        if key == "exploration_fraction" and "exploration_fraction" not in info["result"]["config"]: #hack ray 0.7.3 will have exploration_fraction but not versions later than ~0.9
                            field_val = info["result"]["config"]["exploration_config"]["epsilon_timesteps"] / hacky_timesteps_total # convert to fraction to be similar to old exploration_fraction
                        else:
                            field_val = info["result"]["config"][key]
                        str_to_write = '%.2e' % field_val if isinstance(field_val, float) else str(field_val).replace(' ', '')
                        str_to_write += ' '
                        fout.write(str_to_write)
                elif config_type == "model":
                    # if key == 'conv_filters':
                    fout.write(str(info["result"]["config"]["model"][key]).replace(' ', '') + ' ')

        # Write train stats
        timesteps_total = info["result"]["timesteps_total"] # also has episodes_total and training_iteration
        episode_reward_mean = info["result"]["episode_reward_mean"] # also has max and min
        # print("Custom_metrics: ", info["result"]["step_reward_mean"], info["result"]["step_reward_max"], info["result"]["step_reward_min"])
        episode_len_mean = info["result"]["episode_len_mean"]

        fout.write(str(timesteps_total) + ' ' + '%.2e' % episode_reward_mean +
                ' ' + '%.2e' % episode_len_mean + '\n') # timesteps_total always HAS to be the 1st written: analysis.py depends on it
        fout.close()

        # print("##### hack_filename: ", hack_filename)
        # print(os.getcwd())

        # We did not manage to find an easy way to log evaluation stats for Ray without the following hack which demarcates the end of a training iteration in the evaluation stats file
        hack_filename_eval = stats_file_name + '_eval.csv'
        fout = open(hack_filename_eval, 'a') #hardcoded

        import os, psutil
        mem_used_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        fout.write('#HACK STRING EVAL, mem_used_mb: ' + str(mem_used_mb) + "\n")
        fout.close()

        info["result"]["callback_ok"] = True


    # Ray callback to write evaluation stats to CSV file at end of every training iteration
    # on_episode_end is used because these results won't be available on_train_result but only after every episode has ended during evaluation (evaluation phase is checked for by using dummy_eval)
    def on_episode_end(info):
        if "dummy_eval" in info["env"].get_unwrapped()[0].config:
            # print("###on_episode_end info", info["env"].get_unwrapped()[0].config["make_denser"], info["episode"].total_reward, info["episode"].length) #, info["episode"]._agent_reward_history)
            reward_this_episode = info["episode"].total_reward
            length_this_episode = info["episode"].length
            hack_filename_eval = stats_file_name + '_eval.csv'
            fout = open(hack_filename_eval, 'a') #hardcoded
            fout.write('%.2e' % reward_this_episode + ' ' + str(length_this_episode) + "\n")
            fout.close()

    def on_episode_step(info):
        episode = info["episode"]
        if "step_reward" not in episode.custom_metrics:
            episode.custom_metrics["step_reward"] = []
            step_reward =  episode.total_reward
        else:
            step_reward =  episode.total_reward - np.sum(episode.custom_metrics["step_reward"])
            episode.custom_metrics["step_reward"].append(step_reward) # This line should not be executed the 1st time this function is called because no step has actually taken place then (Ray 0.9.0)!!
        # episode.custom_metrics = {}
        # episode.user_data = {}
        # episode.hist_data = {}
        # Next 2 are the same, except 1st one is total episodic reward _per_ agent
        # episode.agent_rewards = defaultdict(float)
        # episode.total_reward += reward
        # only hack to get per step reward seems to be to store prev total_reward and subtract it from that
        # episode._agent_reward_history[agent_id].append(reward)



    # value_tuples = []
    # for config_type, config_dict in config.var_configs.items():
    #     for key in config_dict:
    #         assert type(config.var_configs[config_type][key]) == list, "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
    #         value_tuples.append(config.var_configs[config_type][key])
    #
    # import itertools
    # cartesian_product_configs = list(itertools.product(*value_tuples))
    # print("Total number of configs. to run:", len(cartesian_product_configs))

    if args.config_num is None:
        cartesian_product_configs = config.cartesian_product_configs
        # pass
    else:
        cartesian_product_configs = [config.cartesian_product_configs[args.config_num]]

    #hack ###TODO remove?
    if 'random_configs' in dir(config):
        random_configs = config.random_configs # [args.agent_config_num]
        if args.agent_config_num is not None:
            random_configs = [config.random_configs[args.agent_config_num]]
            # print("Single random_config of agent to be run:", random_config)
    else:
        random_configs = [()] * len(cartesian_product_configs)



    from functools import reduce
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

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    for enum_conf_1, current_config_ in enumerate(cartesian_product_configs):
        for enum_conf_2, random_config_ in enumerate(random_configs):
            print("current_config of agent to be run:", current_config_, enum_conf_1)
            print("random_config of agent to be run:", random_config_, enum_conf_2)

            current_config = current_config_ + random_config_ ###TODO Remove #hack because this ignores model_config that may be at end of current_config!
            algorithm = config.algorithm

            agent_config = config.agent_config
            model_config = config.model_config
            env_config = config.env_config
            # sys.exit(0)

            for config_type, config_dict in config.var_configs.items():
                for key in config_dict:
                # if config_type == "env_config": # There is a dummy seed in the env_config because it's not used in the environment. It implies a different seed for the agent on every launch as the seed for Ray is not being set here. I faced problems with Ray's seeding process.
                    if config_type == "env": #hack All these are hacks to get around different limitations
                        if key == 'reward_noise':
                            reward_noise_ = current_config[list(var_env_configs).index(key)] # this works because env_configs are 1st in the OrderedDict
                            env_config["env_config"][key] = tune.function(lambda a: a.normal(0, reward_noise_))
                            env_config["env_config"]['reward_noise_std'] = reward_noise_ #hack Needed to be able to write scalar value of std dev. to stats file instead of the lambda function above ###TODO Could remove the hack by creating a class for the noises and changing its repr()
                        elif key == 'transition_noise' and env_config["env_config"]["state_space_type"] == "continuous":
                            transition_noise_ = current_config[list(var_env_configs).index(key)]
                            env_config["env_config"][key] = tune.function(lambda a: a.normal(0, transition_noise_))
                            env_config["env_config"]['transition_noise_std'] = transition_noise_ #hack
                        elif key == 'reward_dist_end_pts':
                            reward_dist_ = current_config[list(var_env_configs).index(key)] # this works because env_configs are 1st in the OrderedDict

                            permutations = []
                            length = current_config[list(var_env_configs).index("sequence_length")]
                            diameter = current_config[list(var_env_configs).index("diameter")]
                            action_space_size = current_config[list(var_env_configs).index("action_space_size")]
                            state_space_size = action_space_size * diameter
                            terminal_state_density = current_config[list(var_env_configs).index("terminal_state_density")]
                            maximum = state_space_size  * (1 - terminal_state_density)/diameter
                            fraction = current_config[list(var_env_configs).index("reward_density")]

                            for i in range(length):
                                permutations.append(maximum - i // diameter)
                            num_possible_permutations = np.prod(permutations)
                            num_sel_sequences = int(fraction * num_possible_permutations)
                            if num_sel_sequences == 0:
                                num_sel_sequences = 1
                                warnings.warn('0 rewardable sequences per independent set for given reward_density, sequence_length, diameter and terminal_state_density. Setting it to 1.')

                            num_rews = diameter * num_sel_sequences
                            print("num_rewardable_sequences set to:", num_rews)
                            if num_rews == 1:
                                rews = [1.0]
                            else:
                                rews = np.linspace(reward_dist_[0], reward_dist_[1], num=num_rews)
                            assert rews[-1] == 1.0
                            np.random.shuffle(rews)

                            def get_rews(rng, r_dict):
                                return rews[len(r_dict)]

                            env_config["env_config"]['reward_dist'] = tune.function(get_rews)
                            env_config["env_config"]['reward_dist_end_pts'] = reward_dist_
                            env_config["env_config"]['state_space_size'] = action_space_size * diameter
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
            if "model" in model_config and model_config["model"]["use_lstm"]:
                model_config["model"]["max_seq_len"] = env_config["env_config"]["delay"] + env_config["env_config"]["sequence_length"] + 1

            if algorithm == 'DDPG': ###TODO Find a better way to enforce these?? Especially problematic for TD3 because then more values for target_noise_clip are witten to CSVs than actually used during HPO but for normal (non-HPO) runs this needs to be not done.
                agent_config["actor_lr"] = agent_config["critic_lr"]
                agent_config["actor_hiddens"] = agent_config["critic_hiddens"]
            elif algorithm == 'TD3':
                agent_config["target_noise_clip"] = agent_config["target_noise_clip_relative"] * agent_config["target_noise"]
                del agent_config["target_noise_clip_relative"] #hack have to delete it otherwise Ray will crash for unknown config param.

            # else: #if algorithm == 'SAC':
            if "state_space_type" in env_config:
                if env_config["env_config"]["state_space_type"] == 'continuous':
                    env_config["env_config"]["action_space_dim"] = env_config["env_config"]["state_space_dim"]

            #hacks end

            eval_config = config.eval_config

            if env_config["env"] in ["HalfCheetahWrapper-v3", "HopperWrapper-v3", "PusherWrapper-v2", "ReacherWrapper-v2"]:
                if "time_unit" in env_config["env_config"]: #hack This is needed so that the environment runs the same amount of seconds of simulation, even though episode steps are different.
                    env_config["horizon"] /= env_config["env_config"]["time_unit"]
                    env_config["horizon"] = int(env_config["horizon"])

                    agent_config["learning_starts"] /= env_config["env_config"]["time_unit"]
                    agent_config["learning_starts"] = int(agent_config["learning_starts"])

                    agent_config["timesteps_per_iteration"] /= env_config["env_config"]["time_unit"]
                    agent_config["timesteps_per_iteration"] = int(agent_config["timesteps_per_iteration"])

                    eval_config["evaluation_config"]["horizon"] /= env_config["env_config"]["time_unit"]
                    eval_config["evaluation_config"]["horizon"] = int(eval_config["evaluation_config"]["horizon"])

                    agent_config["train_batch_size"] *= env_config["env_config"]["time_unit"] # this is needed because Ray (until version 0.8.6 I think) fixes the ratio of number of samples trained/number of steps sampled in environment
                    agent_config["train_batch_size"] = int(agent_config["train_batch_size"])

            extra_config = {
                "callbacks": {
        #                 "on_episode_start": tune.function(on_episode_start),
                    # "on_episode_step": tune.function(on_episode_step),
                    "on_episode_end": tune.function(on_episode_end),
        #                 "on_sample_end": tune.function(on_sample_end),
                    "on_train_result": tune.function(on_train_result),
        #                 "on_postprocess_traj": tune.function(on_postprocess_traj),
                        },
                # "log_level": 'WARN',
            }

            # tune_config = reduce(deepmerge, [agent_config, env_config, model_config, eval_config, extra_config])
            tune_config = {**agent_config, **model_config, **env_config, **eval_config, **extra_config} # This works because the dictionaries involved have mutually exclusive sets of keys, otherwise we would need to use a deepmerge!
            print("tune_config:",)
            pp.pprint(tune_config)

            #hack Common #mujoco wrapper to allow Mujoco envs to be wrapped by MujocoEnvWrapper (which fiddles with lower-level Mujoco stuff) and then by GymEnvWrapper which is more general and basically adds dimensions from MDPP which are common to discrete and continuous environments
            if env_config["env"] in ["HalfCheetahWrapper-v3", "HopperWrapper-v3", "PusherWrapper-v2", "ReacherWrapper-v2"]:
                def create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env):
                    '''Creates a GymEnvWrapper around a MujocoEnvWrapper
                    '''
                    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
                    me = wrapped_mujoco_env(**config)
                    gew = GymEnvWrapper(me, **config) ##IMP Had initially thought to put this config in config["GymEnvWrapper"] but because of code below which converts var_env_configs to env_config, it's best to leave those configs as top level configs in the dict!
                    return gew


            #default settings for #timesteps_total
            if env_config["env"] in ["HalfCheetahWrapper-v3"]: #hack
                timesteps_total = 3000000

                from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
                from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
                wrapped_mujoco_env = get_mujoco_wrapper(HalfCheetahEnv)
                register_env("HalfCheetahWrapper-v3", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

            elif env_config["env"] in ["HopperWrapper-v3"]: #hack
                timesteps_total = 1000000

                from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
                from gym.envs.mujoco.hopper_v3 import HopperEnv
                wrapped_mujoco_env = get_mujoco_wrapper(HopperEnv)
                register_env("HopperWrapper-v3", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

            elif env_config["env"] in ["PusherWrapper-v2"]: #hack
                timesteps_total = 500000

                from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
                from gym.envs.mujoco.pusher import PusherEnv
                wrapped_mujoco_env = get_mujoco_wrapper(PusherEnv)
                register_env("PusherWrapper-v2", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

            elif env_config["env"] in ["ReacherWrapper-v2"]: #hack
                timesteps_total = 500000

                from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
                from gym.envs.mujoco.reacher import ReacherEnv
                wrapped_mujoco_env = get_mujoco_wrapper(ReacherEnv)
                register_env("ReacherWrapper-v2", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

            elif env_config["env"] in ["GymEnvWrapper-Atari"]: #hack
                if "AtariEnv" in env_config["env_config"]:
                    timesteps_total = 10_000_000

            else:
                if algorithm == 'DQN':
                    timesteps_total = 20000
                elif algorithm == 'A3C': #hack
                    timesteps_total = 150000
                else: #if algorithm == 'DDPG': #hack
                    timesteps_total = 20000

            if 'timesteps_total' in dir(config):
                timesteps_total = config.timesteps_total

            if env_config["env"] in ["HalfCheetahWrapper-v3", "HopperWrapper-v3", "PusherWrapper-v2", "ReacherWrapper-v2"]:
                if "time_unit" in env_config["env_config"]: #hack This is needed so that the environment runs the same amount of seconds of simulation, even though episode steps are different.
                    timesteps_total /= env_config["env_config"]["time_unit"]
                    timesteps_total = int(timesteps_total)


            print("\n\033[1;32m======== Running on environment: " + env_config["env"] + " =========\033[0;0m\n")
            print("\n\033[1;32m======== for " + str(timesteps_total) + " steps =========\033[0;0m\n")

            tune.run(
                algorithm,
                name=algorithm + str(args.exp_name.split('/')[-1]) + '_' + str(args.config_num), ####IMP Name has to be specified otherwise, may lead to clashing for temp file in ~/ray_results/... directory.
                stop={
                    "timesteps_total": timesteps_total,
                    },
                config=tune_config,
                checkpoint_at_end=args.save_model,
                local_dir=args.framework_dir + '/_ray_results',
                #return_trials=True # add trials = tune.run( above
            )

    end = time.time()
    print("No. of seconds to run:", end - start)

if __name__ == '__main__':
    args = parse_args()
    main(args)