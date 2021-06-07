from functools import reduce
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import OneHotPreprocessor
import warnings
import numpy as np
from mdp_playground.envs import RLToyEnv
import mdp_playground
from ray.tune.registry import register_env
import copy
import sys
import os

mujoco_envs = [
    "HalfCheetahWrapper-v3",
    "HopperWrapper-v3",
    "PusherWrapper-v2",
    "ReacherWrapper-v2",
]


ModelCatalog.register_custom_preprocessor("ohe", OneHotPreprocessor)


# def init_ray(log_level=None, tmp_dir=None, include_webui=None,
#              object_store_memory=int(2e9),
#              redis_max_memory=int(1e9), local_mode=False):
def init_ray(**kwargs):
    import ray

    if ray.__version__[0] == "1":  # new version 1.0 API
        if "redis_max_memory" in kwargs:
            value = kwargs["redis_max_memory"]
            del kwargs["redis_max_memory"]
            kwargs["_redis_max_memory"] = value
        if "tmp_dir" in kwargs:
            value = kwargs["tmp_dir"]
            del kwargs["tmp_dir"]
            kwargs["_temp_dir"] = value

    else:
        if "tmp_dir" in kwargs:
            value = kwargs["tmp_dir"]
            del kwargs["tmp_dir"]
            kwargs["temp_dir"] = value

    if "log_level" in kwargs:
        value = kwargs["log_level"]
        del kwargs["log_level"]
        kwargs["logging_level"] = value

    ray.init(**kwargs)


def process_configs(
    config_file,
    stats_file_prefix,
    config_num,
    log_level,
    framework="ray",
    framework_dir="/tmp/ray",
):
    config_file_path = os.path.abspath("/".join(config_file.split("/")[:-1]))

    sys.path.insert(1, config_file_path)  # #hack
    import importlib

    config = importlib.import_module(config_file.split("/")[-1], package=None)
    print("Number of seeds for environment:", config.num_seeds)

    # #hacks needed to setup Ray callbacks below
    # #hack because these need to be read in on_train_result and trying to read
    # config there raises an error because it's been imported from a Python
    # module and I think they try to reload the module there.
    variable_configs = []
    if "var_configs" in dir(config):
        variable_configs.append(copy.deepcopy(config.var_configs))

    if "random_configs" in dir(config):
        variable_configs.append(copy.deepcopy(config.random_configs))
    if "sobol_configs" in dir(config):
        variable_configs.append(copy.deepcopy(config.sobol_configs))

    # overwrite = False because the keys in different modes of config generation
    # need to be disjoint
    variable_configs_deepcopy = deepmerge_multiple_dicts(
        *variable_configs, overwrite=False
    )

    if "timesteps_total" in dir(config):
        hacky_timesteps_total = config.timesteps_total  # hack

    config_algorithm = config.algorithm  # hack
    # sys.exit(0)

    columns_to_write = []
    for config_type, config_dict in variable_configs_deepcopy.items():
        for key in config_dict:
            columns_to_write.append(key)

    stats_file_name = stats_file_prefix + ".csv"

    init_stats_file(stats_file_name, columns_to_write)

    # Ray specific setup:
    if framework.lower() == "ray":
        from ray import tune

        setup_ray(config, config_num, log_level, framework_dir)
        on_train_result, on_episode_end = setup_ray_callbacks(
            stats_file_prefix,
            variable_configs_deepcopy,
            hacky_timesteps_total,
            config_algorithm,
        )

        # default Define default config which gets overwritten with config in config.py file if present.
        default_config = {
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

    # Stable Baselines specific setup:
    elif framework.lower() == "stable_baselines":
        ...

    else:
        raise ValueError(
            "Framework passed was not a valid option. It was: "
            + framework
            + ". Available options are: ray and stable_baselines."
        )

    varying_configs = []
    separate_var_configs = []
    # ###IMP Currently num_configs has to be equal for all 3 cases below:
    # grid (i.e. var), random and sobol #TODO Not sure how to solve this #config
    # setup problem. Could take Cartesian product of all 3 but that may lead to
    # too many configs and Cartesian product of dicts is a pain.
    if "var_configs" in dir(config):
        separate_var_configs.append(
            get_list_of_varying_configs(config.var_configs, mode="grid")
        )
    if "sobol_configs" in dir(config):
        separate_var_configs.append(
            get_list_of_varying_configs(
                config.sobol_configs, mode="sobol", num_configs=config.num_configs
            )
        )
    if "random_configs" in dir(config):
        separate_var_configs.append(
            get_list_of_varying_configs(
                config.random_configs, mode="random", num_configs=config.num_configs
            )
        )
    # print("VARYING_CONFIGS:", varying_configs)

    num_configs_ = max(
        [len(separate_var_configs[i]) for i in range(len(separate_var_configs))]
    )
    for i in range(num_configs_):
        to_combine = [
            separate_var_configs[j][i] for j in range(len(separate_var_configs))
        ]
        # overwrite = False because the keys in different modes of
        # config generation need to be disjoint
        varying_configs.append(deepmerge_multiple_dicts(*to_combine, overwrite=False))

    # varying_configs is a list of dict of dicts with a specific structure.
    final_configs = combined_processing(
        default_config,
        config.env_config,
        config.agent_config,
        config.model_config,
        config.eval_config,
        varying_configs=varying_configs,
        framework=framework,
        algorithm=config.algorithm,
    )

    return config, final_configs


def setup_ray(config, config_num, log_level, framework_dir):
    tmp_dir = framework_dir + "/tmp_" + str(config_num)
    # import ray
    if config.algorithm == "DQN":  # hack
        init_ray(log_level=log_level, tmp_dir=tmp_dir)
        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9),
        #          temp_dir=tmp_dir,
        #          logging_level=log_level,
        #          # local_mode=True,
        #          # webui_host='0.0.0.0'); logging_level=logging.INFO,
        #          )
        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9),
        # local_mode=True, plasma_directory='/tmp') #, memory=int(8e9),
        # local_mode=True # local_mode (bool): If true, the code will be executed
        # serially. This is useful for debugging. # when true on_train_result and
        # on_episode_end operate in the same current directory as the script. A3C
        # is crashing in local mode, so didn't use it and had to work around by
        # giving full path + filename in stats_file_name.; also has argument
        # driver_object_store_memory=, plasma_directory='/tmp'
    elif config.algorithm == "A3C":  # hack
        init_ray(log_level=log_level, tmp_dir=tmp_dir)
        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9),
        #          temp_dir=tmp_dir,
        #          logging_level=log_level,
        #          # local_mode=True,
        #          # webui_host='0.0.0.0'); logging_level=logging.INFO,
        #          )        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9), local_mode=True, plasma_directory='/tmp')
    else:
        init_ray(log_level=log_level, tmp_dir=tmp_dir, local_mode=True)

        # ray.init(object_store_memory=int(2e9), redis_max_memory=int(1e9),
        #          temp_dir=tmp_dir,
        #          logging_level=log_level,
        #          local_mode=True,
        #          # webui_host='0.0.0.0'); logging_level=logging.INFO,
        #          )


def init_stats_file(stats_file_name, columns_to_write):
    fout = open(stats_file_name, "a")  # hardcoded
    fout.write("# training_iteration, algorithm, ")
    for column in columns_to_write:
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
        fout.write(column + ", ")
    fout.write(
        "timesteps_total, episode_reward_mean, episode_len_mean\n"
    )  # , mem_used_mb
    fout.close()


def setup_ray_callbacks(
    stats_file_prefix,
    variable_configs_deepcopy,
    hacky_timesteps_total,
    config_algorithm,
):
    # Setup Ray callbacks
    # Ray callback to write training stats to CSV file at end of every training iteration
    # hack Didn't know how to move this function to config. It requires the
    # filename which _has_ to be possible to set in run_experiments.py. Had to
    # take care of stats_file_prefix, variable_configs_deepcopy,
    # hacky_timesteps_total, config_algorithm; and had to initialise file
    # writing in here (config_processor).
    def on_train_result(info):
        training_iteration = info["result"]["training_iteration"]
        # algorithm = info["trainer"]._name

        # Writes every iteration, would slow things down. #hack
        fout = open(stats_file_prefix + ".csv", "a")  # hardcoded
        fout.write(str(training_iteration) + " " + config_algorithm + " ")
        for config_type, config_dict in variable_configs_deepcopy.items():
            for key in config_dict:
                if config_type == "env":
                    field_val = info["result"]["config"]["env_config"][key]
                    if isinstance(field_val, float):
                        str_to_write = "%.2e" % field_val
                    elif isinstance(field_val, list):
                        str_to_write = "["
                        for elem in field_val:
                            # print(key)
                            str_to_write += (
                                "%.2e" % elem if isinstance(elem, float) else str(elem)
                            )
                            str_to_write += ","
                        str_to_write += "]"
                    else:
                        str_to_write = str(field_val).replace(" ", "")
                    str_to_write += " "
                    fout.write(str_to_write)
                elif config_type == "agent":
                    if config_algorithm == "SAC" and key == "critic_learning_rate":
                        real_key = "lr"  # hack due to Ray's weird ConfigSpaces
                        fout.write(
                            "%.2e"
                            % info["result"]["config"]["optimization"][key].replace(
                                " ", ""
                            )
                            + " "
                        )
                    elif config_algorithm == "SAC" and key == "fcnet_hiddens":
                        # hack due to Ray's weird ConfigSpaces
                        str_to_write = (
                            str(info["result"]["config"]["Q_model"][key]).replace(
                                " ", ""
                            )
                            + " "
                        )
                        fout.write(str_to_write)
                    # elif config_algorithm == 'SAC' and key == "policy_model":
                    #     #hack due to Ray's weird ConfigSpaces
                    #     pass
                    # fout.write(str(info["result"]["config"][key]['fcnet_hiddens']).replace(' ', '') + ' ')
                    else:
                        if (
                            key == "exploration_fraction"
                            and "exploration_fraction" not in info["result"]["config"]
                        ):  # hack ray 0.7.3 will have exploration_fraction but not versions later than ~0.9
                            field_val = (
                                info["result"]["config"]["exploration_config"][
                                    "epsilon_timesteps"
                                ]
                                / hacky_timesteps_total
                            )  # convert to fraction to be similar to old exploration_fraction
                        else:
                            field_val = info["result"]["config"][key]
                        str_to_write = (
                            "%.2e" % field_val
                            if isinstance(field_val, float)
                            else str(field_val).replace(" ", "")
                        )
                        str_to_write += " "
                        fout.write(str_to_write)
                elif config_type == "model":
                    # if key == 'conv_filters':
                    fout.write(
                        str(info["result"]["config"]["model"][key]).replace(" ", "")
                        + " "
                    )

        # Write train stats
        timesteps_total = info["result"][
            "timesteps_total"
        ]  # also has episodes_total and training_iteration
        episode_reward_mean = info["result"][
            "episode_reward_mean"
        ]  # also has max and min
        # print("Custom_metrics: ", info["result"]["step_reward_mean"], info["result"]["step_reward_max"], info["result"]["step_reward_min"])
        episode_len_mean = info["result"]["episode_len_mean"]

        # ##TODO write CSV stats configs only once in each case, write runtime and memory, td_error - check tempoRL logs;
        # import os, psutil
        # mem_used_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        fout.write(
            str(timesteps_total)
            + " "
            + "%.2e" % episode_reward_mean
            + " "
            + "%.2e" % episode_len_mean  # + ' ' + '%.2e' % mem_used_mb
            + "\n"
        )  # timesteps_total always HAS to be the 1st written: analysis.py depends on it
        fout.close()

        # print("##### stats_file_name: ", stats_file_name)
        # print(os.getcwd())

        # We did not manage to find an easy way to log evaluation stats for Ray
        # without the following hack which demarcates the end of a training
        # iteration in the evaluation stats file
        if info["result"]["config"]["evaluation_interval"] is not None:
            stats_file_eval = stats_file_prefix + "_eval.csv"
            fout = open(stats_file_eval, "a")  # hardcoded

            fout.write("#HACK STRING EVAL\n")
            fout.close()

        info["result"]["callback_ok"] = True

    # Ray callback to write evaluation stats to CSV file at end of every training iteration
    # on_episode_end is used because these results won't be available on_train_result
    # but only after every episode has ended during evaluation (evaluation phase is
    # checked for by using dummy_eval)
    def on_episode_end(info):
        if "dummy_eval" in info["env"].get_unwrapped()[0].config:
            # print("###on_episode_end info",
            # info["env"].get_unwrapped()[0].config["make_denser"],
            # info["episode"].total_reward, info["episode"].length) #,
            # info["episode"]._agent_reward_history)
            reward_this_episode = info["episode"].total_reward
            length_this_episode = info["episode"].length
            stats_file_eval = stats_file_prefix + "_eval.csv"
            fout = open(stats_file_eval, "a")  # hardcoded
            fout.write(
                "%.2e" % reward_this_episode + " " + str(length_this_episode) + "\n"
            )
            fout.close()

    def on_episode_step(info):
        episode = info["episode"]
        if "step_reward" not in episode.custom_metrics:
            episode.custom_metrics["step_reward"] = []
            step_reward = episode.total_reward
        else:
            step_reward = episode.total_reward - np.sum(
                episode.custom_metrics["step_reward"]
            )
            episode.custom_metrics["step_reward"].append(step_reward)  # This line
            # should not be executed the 1st time this function is called because
            # no step has actually taken place then (Ray 0.9.0)!!
        # episode.custom_metrics = {}
        # episode.user_data = {}
        # episode.hist_data = {}
        # Next 2 are the same, except 1st one is total episodic reward _per_ agent
        # episode.agent_rewards = defaultdict(float)
        # episode.total_reward += reward
        # only hack to get per step reward seems to be to store prev total_reward
        # and subtract it from that
        # episode._agent_reward_history[agent_id].append(reward)

    return on_train_result, on_episode_end


def get_list_of_varying_configs(var_configs, mode="grid", num_configs=None):
    """
    var_configs: dict of dicts of lists as values
        A dict of dicts with lists as the leaf values to allow each
        configuration option to take multiple possible values.
    """

    if mode == "grid":
        varying_configs = get_grid_of_configs(var_configs)

    elif mode == "random":
        varying_configs = get_random_configs(var_configs, num_configs=num_configs)
    elif mode == "sobol":
        varying_configs = sobol_configs_from_config_dict(
            var_configs, num_configs=num_configs
        )

    list_of_configs = []

    for enum_conf_1, current_config in enumerate(varying_configs):
        env_config = {"env": {}}
        model_config = {"model": {}}
        agent_config = {"agent": {}}

        for config_type, config_dict in var_configs.items():
            for key in config_dict:
                # if config_type == "env_config": # There is a dummy seed in the
                # env_config because it's not used in the environment. It implies a
                # different seed for the agent on every launch as the seed for Ray is not
                # being set here. I faced problems with Ray's seeding process.
                if config_type == "env":
                    env_config["env"][key] = current_config[
                        list(var_configs["env"]).index(key)
                    ]

                elif (
                    config_type == "agent"
                ):  # hack All these are hacks to get around different limitations
                    num_configs_done = len(list(var_configs["env"]))
                    agent_config["agent"][key] = current_config[
                        num_configs_done + list(var_configs[config_type]).index(key)
                    ]

                elif config_type == "model":
                    num_configs_done = len(list(var_configs["env"])) + len(
                        list(var_configs["agent"])
                    )
                    model_config["model"][key] = current_config[
                        num_configs_done + list(var_configs[config_type]).index(key)
                    ]

        combined_config = {**agent_config, **model_config, **env_config}

        list_of_configs.append(combined_config)

    return list_of_configs


def get_grid_of_configs(var_configs):
    value_tuples = []

    # #TODO Currently, the var_configs dict is nested, might want to make it
    # single level. However, the config dicts used in Ray are nested, so keep it
    # like this for now. Further, the 2nd level division chosen for configs
    # currently, i.e., env, agent, model is a bit arbitrary, but better like
    # this since it can be compliant with Ray and other frameworks and additional
    # processing can take place in framework_specific_processing() below.
    for config_type, config_dict in var_configs.items():
        for key in config_dict:
            assert (
                isinstance(var_configs[config_type][key], list)
            ), "var_configs should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
            value_tuples.append(var_configs[config_type][key])

    import itertools

    if len(value_tuples) == 0:
        cartesian_product_configs = []  # Edge case, else it'd become [()].
    else:
        cartesian_product_configs = list(itertools.product(*value_tuples))

    print("Total number of configs. to run:", len(cartesian_product_configs))

    return cartesian_product_configs


def get_random_configs(var_configs, num_configs):
    """
    var_configs: dict of dicts of strings or tuples as values
        A dict of dicts with strings or tuples as the leaf values which encode
        a ConfigSpace.

    #TODO Currently num_configs is fixed for each config_type (env, agent or
    model) for this and get_sobol_configs()
    """

    random_configs = []

    for config_type, config_dict in var_configs.items():
        if len(config_dict) > 0:
            cs = create_config_space_from_config_dict(config_dict)
            print("Variable ConfigSpace:")
            print(cs)
            random_configs += cs.sample_configuration(size=num_configs)
            # print("type(random_configs):", type(random_configs))
        else:
            pass

    for i in range(len(random_configs)):
        # if random_configs[i].get_dictionary()["train_batch_size"] == 4 \
        # and random_configs[i].get_dictionary()["buffer_size"] < 33:
        #     print("Config:", i, "train_batch_size, buffer_size:", random_configs[i].get_dictionary()["train_batch_size"], random_configs[i].get_dictionary()["buffer_size"])
        # hack ####TODO Change run_experiments.py and here to directly pass whole
        # config dict to run_experiments.py. Would need to replace in every
        # config.py file.
        random_configs[i] = tuple(random_configs[i].get_dictionary().values())
    print(random_configs[0])
    print(random_configs[1])

    return random_configs


def create_config_space_from_config_dict(config_dict):
    """ """
    import ConfigSpace as CS

    cs = CS.ConfigurationSpace(seed=1234)  # #seed #random
    import ConfigSpace.hyperparameters as CSH
    import json

    for key in config_dict:
        val = config_dict[key]
        if "int" in val:
            lower = int(val.split("[")[1].split(",")[0].strip())
            upper = int(val.split("]")[0].split(",")[-1].strip())
            log = True if "log" in val else False
            cs.add_hyperparameter(
                CSH.UniformIntegerHyperparameter(
                    name=key, lower=lower, upper=upper, log=log
                )
            )
        elif "float" in val:
            lower = float(val.split("[")[1].split(",")[0].strip())
            upper = float(val.split("]")[0].split(",")[-1].strip())
            log = True if "log" in val else False
            cs.add_hyperparameter(
                CSH.UniformFloatHyperparameter(
                    name=key, lower=lower, upper=upper, log=log
                )
            )
        elif "cat" in val:
            # Seems faster than ast.literal_eval (See
            # https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list)
            choices = json.loads("[" + val.split("[")[1].split("]")[0] + "]")
            cs.add_hyperparameter(
                CSH.CategoricalHyperparameter(name=key, choices=choices)
            )
            # print(type(CSH.CategoricalHyperparameter(name=key, choices=choices).choices[0]))

    return cs


def sobol_configs_from_config_dict(var_configs, num_configs):
    """ """

    sobol_configs = []

    for config_type, config_dict in var_configs.items():

        num_dims = 0
        for key in config_dict:
            val = config_dict[key]
            if isinstance(val, tuple):  # i.e. a constant value
                pass
            else:  # i.e. a variable value
                num_dims += 1

        print(
            "Generating sobol sequence with "
            + str(num_configs)
            + " and "
            + str(num_dims)
            + " dimensions:"
        )

        from scipy.optimize._shgo_lib.sobol_seq import (
            Sobol,
        )  # Only generates real vectors in range 0 to 1 per dimension
        import json

        sobol_gen = Sobol()
        sobol = sobol_gen.i4_sobol_generate(num_dims, num_configs, skip=0)
        print(sobol)

        for sample in sobol:
            # print(sample)
            sobol_configs.append({})  # new config
            j = 0
            for key in config_dict:
                val = config_dict[key]
                if isinstance(val, tuple):  # i.e. a constant value
                    sobol_configs[-1][key] = val[0]
                # The rest are config spaces for param settings
                elif "int" in val:
                    lower = float(val.split("[")[1].split(",")[0].strip())
                    upper = float(val.split("]")[0].split(",")[-1].strip())
                    log = True if "log" in val else False
                    # TODO log vals
                    sobol_val = lower + (upper - lower) * sample[j]
                    sobol_configs[-1][key] = int(sobol_val)
                    j += 1
                elif "float" in val:
                    lower = float(val.split("[")[1].split(",")[0].strip())
                    upper = float(val.split("]")[0].split(",")[-1].strip())
                    log = True if "log" in val else False
                    if log:
                        lower = np.log(lower)
                        upper = np.log(upper)
                    sobol_val = lower + (upper - lower) * sample[j]
                    if log:
                        sobol_val = np.exp(sobol_val)
                    if key == "reward_dist":
                        sobol_val = [sobol_val, 1.0]
                    sobol_configs[-1][key] = sobol_val
                    j += 1
                elif "cat" in val:
                    # Seems faster than ast.literal_eval (See
                    # https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list)
                    choices = json.loads("[" + val.split("[")[1].split("]")[0] + "]")
                    len_c = len(choices)
                    if (
                        sample[j] == 1.0
                    ):  # TODO remove? Don't know if sobol samples include 1.0
                        sample[j] -= 1e-10
                    index = int(sample[j] * len_c)
                    sobol_configs[-1][key] = choices[index]
                    j += 1

        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)

    for i, conf in enumerate(sobol_configs):
        sobol_configs[i] = tuple(conf.values())  # hack
        # print(conf)
        # pp.pprint(sobol_configs[i])
    print(sobol_configs[0])
    print(sobol_configs[1])

    return sobol_configs


def combined_processing(*static_configs, varying_configs, framework="ray", algorithm):
    """
    varying_configs is a list of dict of dicts with structure of each element
    in the list as: {
        "env": {...}
        "agent": {...}
        "model": {...}
    }
    """
    # print(len(configs))
    # print(type(configs))
    # print(type(*configs))

    # Pre-processing common to frameworks:
    for i, varying_config in enumerate(varying_configs):

        # ###IMP This needs to be done before merging because otherwise
        # varying_config["env"] clashes with "env" key of Ray Tune config later.
        varying_config = {
            "env_config": varying_config["env"],
            **varying_config["agent"],
            "model": varying_config["model"],
        }
        varying_configs[i] = varying_config

    # Ray specific pre-processing
    if framework.lower() == "ray":
        ...

    # Stable Baselines specific pre-processing
    elif framework.lower() == "stable_baselines":
        ...

    else:
        raise ValueError(
            "Framework passed was not a valid option. It was: "
            + framework
            + ". Available options are: ray and stable_baselines."
        )

    # Merge all configs into one
    final_configs = []
    for i in range(len(varying_configs)):
        static_configs_copy = copy.deepcopy(static_configs)
        merged_conf = deepmerge_multiple_dicts(*static_configs_copy, varying_configs[i])
        final_configs.append(merged_conf)

    # Post-processing common to frameworks:
    for i, final_config in enumerate(final_configs):
        if final_configs[i]["env"] in mujoco_envs:
            if "time_unit" in final_configs[i]["env_config"]:  # #hack This is
                # needed so that the environment runs the same amount of seconds of
                # simulation, even though episode steps are different.
                final_configs[i]["horizon"] /= final_configs[i]["env_config"][
                    "time_unit"
                ]
                final_configs[i]["horizon"] = int(final_configs[i]["horizon"])

                final_configs[i]["learning_starts"] /= final_configs[i]["env_config"][
                    "time_unit"
                ]
                final_configs[i]["learning_starts"] = int(
                    final_configs[i]["learning_starts"]
                )

                final_configs[i]["timesteps_per_iteration"] /= final_configs[i][
                    "env_config"
                ]["time_unit"]
                final_configs[i]["timesteps_per_iteration"] = int(
                    final_configs[i]["timesteps_per_iteration"]
                )

                final_configs[i]["evaluation_config"]["horizon"] /= final_configs[i][
                    "env_config"
                ]["time_unit"]
                final_configs[i]["evaluation_config"]["horizon"] = int(
                    final_configs[i]["evaluation_config"]["horizon"]
                )

                final_configs[i]["train_batch_size"] *= final_configs[i]["env_config"][
                    "time_unit"
                ]  # this is needed
                # because Ray (until version 0.8.6 I think) fixes the
                # ratio of number of samples trained/number of steps sampled
                # in environment
                final_configs[i]["train_batch_size"] = int(
                    final_configs[i]["train_batch_size"]
                )

        # #hack Common #mujoco wrapper to allow Mujoco envs to be wrapped by
        # MujocoEnvWrapper (which fiddles with lower-level Mujoco stuff) and
        # then by GymEnvWrapper which is more general and basically adds
        # dimensions from MDPP which are common to discrete and continuous
        # environments

        # if final_configs[i]["env"] in mujoco_envs:

        # #default settings for #timesteps_total
        if final_configs[i]["env"] in ["HalfCheetahWrapper-v3"]:  # hack
            timesteps_total = 3000000

            from mdp_playground.envs.mujoco_env_wrapper import (
                get_mujoco_wrapper,
            )  # hack
            from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

            wrapped_mujoco_env = get_mujoco_wrapper(HalfCheetahEnv)
            register_env(
                "HalfCheetahWrapper-v3",
                lambda config: create_gym_env_wrapper_mujoco_wrapper(
                    config, wrapped_mujoco_env
                ),
            )

        elif final_configs[i]["env"] in ["HopperWrapper-v3"]:  # hack
            timesteps_total = 1000000

            from mdp_playground.envs.mujoco_env_wrapper import (
                get_mujoco_wrapper,
            )  # hack
            from gym.envs.mujoco.hopper_v3 import HopperEnv

            wrapped_mujoco_env = get_mujoco_wrapper(HopperEnv)
            register_env(
                "HopperWrapper-v3",
                lambda config: create_gym_env_wrapper_mujoco_wrapper(
                    config, wrapped_mujoco_env
                ),
            )

        elif final_configs[i]["env"] in ["PusherWrapper-v2"]:  # hack
            timesteps_total = 500000

            from mdp_playground.envs.mujoco_env_wrapper import (
                get_mujoco_wrapper,
            )  # hack
            from gym.envs.mujoco.pusher import PusherEnv

            wrapped_mujoco_env = get_mujoco_wrapper(PusherEnv)
            register_env(
                "PusherWrapper-v2",
                lambda config: create_gym_env_wrapper_mujoco_wrapper(
                    config, wrapped_mujoco_env
                ),
            )

        elif final_configs[i]["env"] in ["ReacherWrapper-v2"]:  # hack
            timesteps_total = 500000

            from mdp_playground.envs.mujoco_env_wrapper import (
                get_mujoco_wrapper,
            )  # hack
            from gym.envs.mujoco.reacher import ReacherEnv

            wrapped_mujoco_env = get_mujoco_wrapper(ReacherEnv)
            register_env(
                "ReacherWrapper-v2",
                lambda config: create_gym_env_wrapper_mujoco_wrapper(
                    config, wrapped_mujoco_env
                ),
            )

        elif final_configs[i]["env"] in ["GymEnvWrapper-Atari"]:  # hack
            if "AtariEnv" in final_configs[i]["env_config"]:
                timesteps_total = 10_000_000

        else:
            if algorithm == "DQN":
                timesteps_total = 20000
            elif algorithm == "A3C":  # hack
                timesteps_total = 150000
            else:  # if algorithm == 'DDPG': #hack
                timesteps_total = 20000

        if final_configs[i]["env"] in mujoco_envs:
            # hack This is needed so that the environment runs the same amount of
            # seconds of simulation, even though episode steps are different.
            if ("time_unit" in final_configs[i]["env_config"]):
                timesteps_total /= final_configs[i]["env_config"]["time_unit"]
                timesteps_total = int(timesteps_total)

        final_configs[i]["timesteps_total"] = timesteps_total

    # Post-processing for Ray:
    if framework.lower() == "ray":
        for i in range(len(final_configs)):
            # for config_type in varying_config:
            for key in final_configs[i]:
                value = final_configs[i][key]

                if algorithm == "SAC":
                    if key == "critic_learning_rate":  # hack
                        final_configs[i]["optimization"] = {
                            key: value,
                            "actor_learning_rate": value,
                            "entropy_learning_rate": value,
                        }
                    if key == "fcnet_hiddens":  # hack
                        final_configs[i]["Q_model"] = {
                            key: value,
                            "fcnet_activation": "relu",
                        }
                        final_configs[i]["policy_model"] = {
                            key: value,
                            "fcnet_activation": "relu",
                        }

                    # TODO Find a better way to enforce these?? Especially problematic for TD3
                    # because then more values for target_noise_clip are witten to CSVs than
                    # actually used during HPO but for normal (non-HPO) runs this needs to be
                    # not done.
                    if (algorithm == "DDPG"):
                        if key == "critic_lr":
                            final_configs[i]["actor_lr"] = value
                        if key == "critic_hiddens":
                            final_configs[i]["actor_hiddens"] = value
                    if algorithm == "TD3":
                        if key == "target_noise_clip_relative":
                            final_configs[i]["target_noise_clip"] = (
                                final_configs[i]["target_noise_clip_relative"]
                                * final_configs[i]["target_noise"]
                            )
                            del final_configs[i][
                                "target_noise_clip_relative"
                            ]  # hack have to delete it otherwise Ray will crash for unknown config param.

                elif key == "model":
                    for key_2 in final_configs[i][key]:
                        if key_2 == "use_lstm":
                            final_configs[i][key]["max_seq_len"] = (
                                final_configs[i]["env_config"]["delay"]
                                + final_configs[i]["env_config"]["sequence_length"]
                                + 1
                            )

    # Post-processing for Stable Baselines:
    elif framework.lower() == "stable_baselines":
        ...

    return final_configs


def create_gym_env_wrapper_mujoco_wrapper(config, wrapped_mujoco_env):
    """Creates a GymEnvWrapper around a MujocoEnvWrapper"""
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper

    me = wrapped_mujoco_env(**config)
    # IMP Had initially thought to put this config in config["GymEnvWrapper"]
    # but because of code below which converts var_env_configs to env_config,
    # it's best to leave those configs as top level configs in the dict!
    gew = GymEnvWrapper(me, **config)
    return gew


def deepmerge_multiple_dicts(*configs, overwrite=True):
    """ """
    merged_configs = {}
    for i in range(len(configs)):
        # print(i)
        merged_configs = deepmerge(merged_configs, configs[i], overwrite=overwrite)

    return merged_configs


def deepmerge(a, b, path=None, overwrite=True):
    """Merges dict b into dict a

    overwrite : bool
        Overwrites value in a with value in b if True with a warning, else raises Exception

    Based on: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries/7205107#7205107
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                if overwrite:
                    warnings.warn(
                        "Overwrote value "
                        + str(a[key])
                        + " with "
                        + str(b[key])
                        + " while merging dicts."
                    )
                    a[key] = b[key]
                else:
                    raise Exception(
                        "Conflict at %s" % ".".join(path + [str(key)])
                        + str(a[key])
                        + " conflicts with "
                        + str(b[key])
                        + " while merging dicts."
                    )
        else:
            a[key] = b[key]
    return a


def post_processing(framework="ray"):
    if framework == "ray":
        import ray

        ray.shutdown()


def create_gym_env_wrapper_atari(config):
    from gym.envs.atari import AtariEnv
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper

    ae = AtariEnv(**config["AtariEnv"])
    # IMP Had initially thought to put this config in config["GymEnvWrapper"]
    # but because of code below which converts var_env_configs to env_config,
    # it's best to leave those configs as top level configs in the dict!
    gew = GymEnvWrapper(ae, **config)
    return gew


def create_gym_env_wrapper_frame_stack_atari(config):  # hack ###TODO remove?
    """When using frameStack GymEnvWrapper should wrap AtariEnv using wrap_deepmind_ray and therefore this function sets "wrap_deepmind_ray": True and 'frame_skip': 1 inside config so as to keep config same as for create_gym_env_wrapper_atari above and reduce manual errors when switching between the 2."""
    config["wrap_deepmind_ray"] = True  # hack
    config["frame_skip"] = 1  # hack
    from gym.envs.atari import AtariEnv
    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    import gym

    game = config["AtariEnv"]["game"]
    game = "".join([g.capitalize() for g in game.split("_")])
    ae = gym.make("{}NoFrameskip-v4".format(game))
    # IMP Had initially thought to put this config in config["GymEnvWrapper"]
    # but because of code below which converts var_env_configs to env_config,
    # it's best to leave those configs as top level configs in the dict!
    gew = GymEnvWrapper(ae, **config)
    return gew


register_env("RLToy-v0", lambda config: RLToyEnv(**config))
register_env("GymEnvWrapper-Atari", lambda config: create_gym_env_wrapper_atari(config))
register_env(
    "GymEnvWrapperFrameStack-Atari",
    lambda config: create_gym_env_wrapper_frame_stack_atari(config),
)
