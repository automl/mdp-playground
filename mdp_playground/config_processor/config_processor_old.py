
from os import stat
from numpy.core.fromnumeric import var
from ray.tune.registry import register_env
from .baselines_processor import agent_to_baselines, model_to_policy_kwargs
import copy

mujoco_envs = ["HalfCheetahWrapper-v3", "HopperWrapper-v3", "PusherWrapper-v2", "ReacherWrapper-v2"]

def get_grid_of_configs(var_configs):
    '''
    var_configs: dict of dicts of lists as values
        A dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values.
    '''
    value_tuples = []

    # TODO Currently, the var_configs dict is nested, might want to make it single level. However, the config dicts used in Ray are nested, so keep it like this for now. Further, the 2nd level division chosen for configs currently, i.e., env, agent, model is a bit arbitrary, but better like this since it can be compliant with Ray and other frameworks and additional processing can take place in framework_specific_processing() below.
    for config_type, config_dict in var_configs.items():
        for key in config_dict:
            assert type(var_configs[config_type][key]) == list, "var_configs should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
            value_tuples.append(var_configs[config_type][key])

    import itertools
    if len(value_tuples) == 0:
        cartesian_product_configs = [] # Edge case, else it'd become [()].
    else:
        cartesian_product_configs = list(itertools.product(*value_tuples))

    print("Total number of configs. to run:", len(cartesian_product_configs))

    grid_of_configs = []

    for enum_conf_1, current_config in enumerate(cartesian_product_configs):
        env_config = {"env": {}}
        model_config = {"model": {}}
        agent_config = {"agent": {}}

        for config_type, config_dict in var_configs.items():
            for key in config_dict:
            # if config_type == "env_config": # There is a dummy seed in the env_config because it's not used in the environment. It implies a different seed for the agent on every launch as the seed for Ray is not being set here. I faced problems with Ray's seeding process.
                if config_type == "env":
                    env_config["env"][key] = current_config[list(var_configs["env"]).index(key)]

                elif config_type == "agent": #hack All these are hacks to get around different limitations
                    num_configs_done = len(list(var_configs["env"]))
                    agent_config["agent"][key] = current_config[num_configs_done + list(var_configs[config_type]).index(key)]

                elif config_type == "model":
                    num_configs_done = len(list(var_configs["env"])) + len(list(var_agent_configs))
                    model_config["model"][key] = current_config[num_configs_done + list(var_configs[config_type]).index(key)]

        combined_config = {**agent_config, **model_config, **env_config}

        grid_of_configs.append(combined_config)

    return grid_of_configs



def combined_processing(*static_configs, varying_configs, framework='ray', algorithm):
    '''
    varying_configs is a dict of dicts with structure: {
    }
    '''
    # print(len(configs))
    # print(type(configs))
    # print(type(*configs))

    # Pre-processing common to frameworks:
    for i, varying_config in enumerate(varying_configs):

        ###IMP This needs to be done before merging because otherwise varying_config["env"] clashes with
        varying_config = {"env_config": varying_config["env"], **varying_config["agent"], "model": varying_config["model"]}
        varying_configs[i] = varying_config

    # Ray specific pre-processing
    if framework.lower() == 'ray':
        ...

    # Stable Baselines specific pre-processing
    elif framework.lower() == 'stable_baselines':
        env_cfg, agent_cfg, model_cfg, eval_cfg = static_configs
        static_configs = list(static_configs)
        # Change variable configuration keys
        for i, cfg_dict in enumerate(varying_configs):
            var_cfg = cfg_dict.copy()
            var_env_cfg = var_cfg.pop('env_config')
            var_model_cfg = var_cfg.pop('model')
            cfg = {"model": var_model_cfg,
                   "agent": var_cfg}
            # make agent and model static empty configs
            curr_cfg = (algorithm, {}, {}, cfg)
            # algorithm, agent_config, model_config, var_configs
            _, _, var_agent_configs, var_model_configs =\
                agent_to_baselines(curr_cfg)
            varying_configs[i] = {}
            varying_configs[i]["env_config"] = var_env_cfg
            varying_configs[i]["model"] = var_model_configs
            varying_configs[i]["agent"] = var_agent_configs
        # Change static configuration keys
        # Make variable empty configs
        curr_cfg = (algorithm, agent_cfg, model_cfg, {})
        agent_config, model_config, _, _ = agent_to_baselines(curr_cfg)
        static_configs = (env_cfg,
                          {"agent": agent_config},
                          {"model": model_config},
                          eval_cfg)
        static_configs = tuple(static_configs)
    else:
        raise ValueError("Framework passed was not a valid option. It was: " + framework + ". Available options are: ray and stable_baselines.")

    # Merge all configs into one
    final_configs = []
    for i in range(len(varying_configs)):
        # for in range(len()):
        static_configs_copy = copy.deepcopy(static_configs)
        merged_conf = deepmerge_multiple_dicts(*static_configs_copy, varying_configs[i])
        final_configs.append(merged_conf) # varying_configs, env_config, agent_config, eval_config

    # Post-processing common to frameworks:
    for i, final_config in enumerate(final_configs):
        if final_configs[i]["env"] in mujoco_envs:
            if "time_unit" in final_configs[i]["env_config"]: #hack This is needed so that the environment runs the same amount of seconds of simulation, even though episode steps are different.
                final_configs[i]["horizon"] /= final_configs[i]["env_config"]["time_unit"]
                final_configs[i]["horizon"] = int(final_configs[i]["horizon"])

                final_configs[i]["learning_starts"] /= final_configs[i]["env_config"]["time_unit"]
                final_configs[i]["learning_starts"] = int(final_configs[i]["learning_starts"])

                final_configs[i]["timesteps_per_iteration"] /= final_configs[i]["env_config"]["time_unit"]
                final_configs[i]["timesteps_per_iteration"] = int(final_configs[i]["timesteps_per_iteration"])

                final_configs[i]["evaluation_config"]["horizon"] /= final_configs[i]["env_config"]["time_unit"]
                final_configs[i]["evaluation_config"]["horizon"] = int(final_configs[i]["evaluation_config"]["horizon"])

                final_configs[i]["train_batch_size"] *= final_configs[i]["env_config"]["time_unit"] # this is needed because Ray (until version 0.8.6 I think) fixes the ratio of number of samples trained/number of steps sampled in environment
                final_configs[i]["train_batch_size"] = int(final_configs[i]["train_batch_size"])

        # hack Common #mujoco wrapper to allow Mujoco envs to be wrapped by MujocoEnvWrapper (which fiddles with lower-level Mujoco stuff) and then by GymEnvWrapper which is more general and basically adds dimensions from MDPP which are common to discrete and continuous environments
        # if final_configs[i]["env"] in mujoco_envs:

        # default settings for #timesteps_total
        if final_configs[i]["env"] in ["HalfCheetahWrapper-v3"]: #hack
            timesteps_total = 3000000

            from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
            from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
            wrapped_mujoco_env = get_mujoco_wrapper(HalfCheetahEnv)
            register_env("HalfCheetahWrapper-v3", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

        elif final_configs[i]["env"] in ["HopperWrapper-v3"]: #hack
            timesteps_total = 1000000

            from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
            from gym.envs.mujoco.hopper_v3 import HopperEnv
            wrapped_mujoco_env = get_mujoco_wrapper(HopperEnv)
            register_env("HopperWrapper-v3", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

        elif final_configs[i]["env"] in ["PusherWrapper-v2"]: #hack
            timesteps_total = 500000

            from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
            from gym.envs.mujoco.pusher import PusherEnv
            wrapped_mujoco_env = get_mujoco_wrapper(PusherEnv)
            register_env("PusherWrapper-v2", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

        elif final_configs[i]["env"] in ["ReacherWrapper-v2"]: #hack
            timesteps_total = 500000

            from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
            from gym.envs.mujoco.reacher import ReacherEnv
            wrapped_mujoco_env = get_mujoco_wrapper(ReacherEnv)
            register_env("ReacherWrapper-v2", lambda config: create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env))

        elif final_configs[i]["env"] in ["GymEnvWrapper-Atari"]: #hack
            if "AtariEnv" in final_configs[i]["env_config"]:
                timesteps_total = 10_000_000

        else:
            if algorithm == 'DQN':
                timesteps_total = 20000
            elif algorithm == 'A3C':  # hack
                timesteps_total = 150000
            else:  # if algorithm == 'DDPG': #hack
                timesteps_total = 20000

        if final_configs[i]["env"] in mujoco_envs:
            if "time_unit" in final_configs[i]["env_config"]: #hack This is needed so that the environment runs the same amount of seconds of simulation, even though episode steps are different.
                timesteps_total /= final_configs[i]["env_config"]["time_unit"]
                timesteps_total = int(timesteps_total)

        final_configs[i]["timesteps_total"] = timesteps_total

    # Post-processing for Ray:
    if framework.lower() == 'ray':
        for i in range(len(final_configs)):
            # for config_type in varying_config:
            for key in final_configs[i]:
                value = final_configs[i][key]

                if algorithm == 'SAC':
                    if key == 'critic_learning_rate': # hack
                        final_configs[i]['optimization'] = {
                                                    key: value,
                                                    'actor_learning_rate': value,
                                                    'entropy_learning_rate': value,
                                                    }
                    if key == 'fcnet_hiddens': #hack
                        final_configs[i]['Q_model'] = {
                                                key: value,
                                                "fcnet_activation": "relu",
                                                }
                        final_configs[i]['policy_model'] = {
                                                key: value,
                                                "fcnet_activation": "relu",
                                                }

                    if algorithm == 'DDPG': ###TODO Find a better way to enforce these?? Especially problematic for TD3 because then more values for target_noise_clip are witten to CSVs than actually used during HPO but for normal (non-HPO) runs this needs to be not done.
                        if key == "critic_lr":
                            final_configs[i]["actor_lr"] = value
                        if key == "critic_hiddens":
                            final_configs[i]["actor_hiddens"] = value
                    if algorithm == 'TD3':
                        if key == "target_noise_clip_relative":
                            final_configs[i]["target_noise_clip"] = final_configs[i]["target_noise_clip_relative"] * final_configs[i]["target_noise"]
                            del final_configs[i]["target_noise_clip_relative"] #hack have to delete it otherwise Ray will crash for unknown config param.

                elif key == "model":
                    for key_2 in final_configs[i][key]:
                        if key_2 == "use_lstm":
                            final_configs[i][key][key_2]["max_seq_len"] = final_configs[i]["env"]["env_config"]["delay"] + final_configs[i]["env"]["env_config"]["sequence_length"] + 1

    # Post-processing for Stable Baselines:
    elif framework.lower() == 'stable_baselines':
        ts_k = "timesteps_per_iteration"
        for i, config in enumerate(final_configs):
            model_config = config["model"]
            use_cnn = False
            if("feature_extraction" in config["agent"]["policy_kwargs"]):
                if config["agent"]["policy_kwargs"]["feature_extraction"] == "cnn":
                    use_cnn = True
            feat_ext = 'mlp' if config["env_config"]["state_space_type"] == "discrete" \
                        and not use_cnn else 'cnn'
            policy_kwargs, use_lstm = \
                model_to_policy_kwargs(feat_ext, model_config)  # return policy_kwargs
            config["agent"]["policy_kwargs"].update(policy_kwargs)
            if(ts_k in config["agent"]):
                # this is not part of baselines parameters, need to separate
                timesteps_per_iteration = config["agent"].pop(ts_k)
                config[ts_k] = timesteps_per_iteration
            else:
                config[ts_k] = 1000  # default
            config["use_lstm"] = use_lstm

    return final_configs


###TODO **extra_config}

def create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env):
    '''Creates a GymEnvWrapper around a MujocoEnvWrapper
    '''
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    me = wrapped_mujoco_env(**config)
    gew = GymEnvWrapper(me, **config) ##IMP Had initially thought to put this config in config["GymEnvWrapper"] but because of code below which converts var_env_configs to env_config, it's best to leave those configs as top level configs in the dict!
    return gew


def deepmerge_multiple_dicts(*configs):
    '''
    '''
    merged_configs = {}
    for i in range(len(configs)):
        # print(i)
        merged_configs = deepmerge(merged_configs, configs[i])

    return merged_configs


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


import mdp_playground
from mdp_playground.envs import RLToyEnv
from ray.tune.registry import register_env
register_env("RLToy-v0", lambda config: RLToyEnv(**config))

def create_gym_env_wrapper_atari(config):
    from gym.envs.atari import AtariEnv
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    ae = AtariEnv(**config["AtariEnv"])
    gew = GymEnvWrapper(ae, **config) ##IMP Had initially thought to put this config in config["GymEnvWrapper"] but because of code below which converts var_env_configs to env_config, it's best to leave those configs as top level configs in the dict!
    return gew

register_env("GymEnvWrapper-Atari", lambda config: create_gym_env_wrapper_atari(config))


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

register_env("GymEnvWrapperFrameStack-Atari", lambda config: create_gym_env_wrapper_frame_stack_atari(config))