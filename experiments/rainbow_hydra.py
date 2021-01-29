num_seeds = 1
timesteps_total = 20_000

import numpy as np
from collections import OrderedDict

var_env_configs = OrderedDict({
    'action_space_size': (8,),#, 10, 12, 14] # [2**i for i in range(1,6)]
    # 'action_space_size': (64),#2, 4, 8, 16] # [2**i for i in range(1,6)]
    'delay': "cat, " + str([i for i in range(11)]), # + [2**i for i in range(4)],
    'sequence_length': "cat, " + str([i for i in range(1, 4)]), #, 2, 3, 4],#i for i in range(1,4)]
    'diameter': "cat, " + str([2**i for i in range(4)]), # + [2**i for i in range(4)],
    'reward_density': "float, [0.03, 0.5]", # np.linspace(0.0, 1.0, num=5)
    'make_denser': (False,),
    'terminal_state_density': (0.25,), # np.linspace(0.1, 1.0, num=5)
    'reward_dist_end_pts': "float, [0.01, 0.8]",
    'reward_scale': "float, log, [0.1, 100]",
    'dummy_seed': (0,), #"cat, " + str([i for i in range(num_seeds)]),
})


print(var_env_configs)
cartesian_product_configs = []
def sobol_configs_from_config_dict(config_dict):
    '''
    '''

    num_prob_inst = 10
    num_dims = 0
    for key in config_dict:
        val = config_dict[key]
        if type(val) == tuple: # i.e. a constant value
            pass
        else: # i.e. a variable value
            num_dims += 1

    print("Generating sobol sequence with " + str(num_prob_inst) + " and " + str(num_dims) + " dimensions:")

    from scipy.optimize._shgo_lib.sobol_seq import Sobol # Only generates real vectors in range 0 to 1 per dimension
    import json
    sobol_gen = Sobol()
    sobol = sobol_gen.i4_sobol_generate(num_dims, num_prob_inst, skip=0)
    print(sobol)

    for sample in sobol:
        # print(sample)
        cartesian_product_configs.append({}) # new config
        j = 0
        for key in config_dict:
            val = config_dict[key]
            if type(val) == tuple: # i.e. a constant value
                cartesian_product_configs[-1][key] = val[0]
            # The rest are config spaces for param settings
            elif "int" in val:
                lower = float(val.split("[")[1].split(",")[0].strip())
                upper = float(val.split("]")[0].split(",")[-1].strip())
                log = True if "log" in val else False
                #TODO log vals
                sobol_val = lower + (upper - lower) * sample[j]
                cartesian_product_configs[-1][key] = int(sobol_val)
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
                if key == "reward_dist_end_pts":
                    sobol_val = [sobol_val, 1.0]
                cartesian_product_configs[-1][key] = sobol_val
                j += 1
            elif "cat" in val:
                choices = json.loads("[" + val.split("[")[1].split("]")[0] + "]") # Seems faster than ast.literal_eval (See https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list)
                len_c = len(choices)
                if sample[j] == 1.0: #TODO remove? Don't know if sobol samples include 1.0
                    sample[j] -= 1e-10
                index = int(sample[j] * len_c)
                cartesian_product_configs[-1][key] = choices[index]
                j += 1



sobol_configs_from_config_dict(var_env_configs)
# import pprint
# pp = pprint.PrettyPrinter(indent=4)

for i, conf in enumerate(cartesian_product_configs):
    cartesian_product_configs[i] = list(conf.values()) #hack
    # print(conf)
    # pp.pprint(cartesian_product_configs[i])

var_agent_configs = OrderedDict({

    "lr": "float, log, [1e-5, 1e-3]", # 1e-4
    "learning_starts": "int, [1, 2000]", # 500
    "target_network_update_freq": "int, log, [1, 2000]", # 800,
    "exploration_fraction": "float, [0.01, 0.99]", # 0.1,
    "n_step": "int, [1, 8]", # 1
    "buffer_size": "int, log, [10, 20000]", # ?? 1000000,
    "adam_epsilon": "float, log, [1e-12, 1e-1]", # ?? 1e-4,
    "train_batch_size": "cat, [4, 8, 16, 32, 64, 128]", # 32,

})

var_agent_configs = OrderedDict(sorted(var_agent_configs.items(), key=lambda t: t[0])) #hack because ConfigSpace below orders alphabetically, the returned configs are in a jumbled order compared to the order above.

def create_config_space_from_config_dict(config_dict):
    '''
    '''
    import ConfigSpace as CS
    cs = CS.ConfigurationSpace(seed=1234)
    import ConfigSpace.hyperparameters as CSH
    import json

    for key in config_dict:
        val = config_dict[key]
        if "int" in val:
            lower = int(val.split("[")[1].split(",")[0].strip())
            upper = int(val.split("]")[0].split(",")[-1].strip())
            log = True if "log" in val else False
            cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name=key, lower=lower, upper=upper, log=log))
        elif "float" in val:
            lower = float(val.split("[")[1].split(",")[0].strip())
            upper = float(val.split("]")[0].split(",")[-1].strip())
            log = True if "log" in val else False
            cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name=key, lower=lower, upper=upper, log=log))
        elif "cat" in val:
            choices = json.loads("[" + val.split("[")[1].split("]")[0] + "]") # Seems faster than ast.literal_eval (See https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list)
            cs.add_hyperparameter(CSH.CategoricalHyperparameter(name=key, choices=choices))
            # print(type(CSH.CategoricalHyperparameter(name=key, choices=choices).choices[0]))

    return cs

cs = create_config_space_from_config_dict(var_agent_configs)
print("Agent variable ConfigSpace:")
print(cs)
num_agent_configs = 10
random_configs = cs.sample_configuration(size=num_agent_configs)
for i in range(len(random_configs)):
    random_configs[i] = list(random_configs[i].get_dictionary().values()) #hack ####TODO Change run_experiments.py and here to directly pass whole config dict to run_experiments.py. Would need to replace in every config.py file.
# print(random_configs)

var_configs = OrderedDict({
"env": var_env_configs,
"agent": var_agent_configs,

})

env_config = {
    "env": "RLToy-v0",
    "horizon": 100,
    "env_config": {
        'seed': 0, #seed
        'state_space_type': 'discrete',
        'action_space_type': 'discrete',
        'generate_random_mdp': True,
        'repeats_in_sequences': False,
        'reward_scale': 1.0,
        'completely_connected': True,
    },
}

algorithm = "DQN"
agent_config = {
    # "adam_epsilon": 1e-4,
    # "buffer_size": 1000000,
    "double_q": True,
    "dueling": True,
    # "lr": 1e-3,
    "exploration_final_eps": 0.01,
    # "exploration_fraction": 0.1,
    "schedule_max_timesteps": 20000,
    # "learning_starts": 500,
    # "target_network_update_freq": 800,
    # "n_step": 4,
    "noisy": True,
    "num_atoms": 10, # [5, 10, 20]
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.75, #
    "prioritized_replay_beta": 0.4,
    "final_prioritized_replay_beta": 1.0, #
    "beta_annealing_fraction": 1.0, #
    # "hiddens": None,

    "sample_batch_size": 4,
    "timesteps_per_iteration": 1000,
    # "train_batch_size": 32,
    "min_iter_time_s": 0,

    # 'num_gpus': 0,
    # "num_workers": 1, # extra workers I think
    # "num_cpus_for_driver": 2,

    "tf_session_args": {
        # note: overriden by `local_tf_session_args`
        "intra_op_parallelism_threads": 4,
        "inter_op_parallelism_threads": 4,
        # "gpu_options": {
        #     "allow_growth": True,
        # },
        # "log_device_placement": False,
        "device_count": {
            "CPU": 2
        },
        # "allow_soft_placement": True,  # required by PPO multi-gpu
    },
    # Override the following tf session args on the local worker
    "local_tf_session_args": {
        "intra_op_parallelism_threads": 4,
        "inter_op_parallelism_threads": 4,
    },

}

model_config = {
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
}

from ray import tune
eval_config = {
    "evaluation_interval": 1, # I think this means every x training_iterations
    "evaluation_config": {
        "explore": False,
        "exploration_fraction": 0,
        "exploration_final_eps": 0,
        "evaluation_num_episodes": 10,
        "horizon": 100,
        "env_config": {
            "dummy_eval": True, #hack Used to check if we are in evaluation mode or training mode inside Ray callback on_episode_end() to be able to write eval stats
            'transition_noise': 0 if "state_space_type" in env_config["env_config"] and env_config["env_config"]["state_space_type"] == "discrete" else tune.function(lambda a: a.normal(0, 0)),
            'reward_noise': tune.function(lambda a: a.normal(0, 0)),
            'action_loss_weight': 0.0,
        }
    },
}

# value_tuples = []
# for config_type, config_dict in var_configs.items():
#     for key in config_dict:
#         assert type(var_configs[config_type][key]) == list, "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
#         value_tuples.append(var_configs[config_type][key])
#
# import itertools
# cartesian_product_configs = list(itertools.product(*value_tuples))
# print("Total number of configs. to run:", len(cartesian_product_configs))
