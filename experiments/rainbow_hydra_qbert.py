num_seeds = 1
timesteps_total = 10_000_000
from collections import OrderedDict

var_env_configs = OrderedDict({
    'delay': [0],
    'dummy_seed': [i for i in range(num_seeds)],
})

var_configs = OrderedDict({
"env": var_env_configs
})

value_tuples = []
for config_type, config_dict in var_configs.items():
    for key in config_dict:
        assert type(var_configs[config_type][key]) == list, "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
        value_tuples.append(var_configs[config_type][key])

import itertools
cartesian_product_configs = list(itertools.product(*value_tuples))
print("Total number of grid configs. to run:", len(cartesian_product_configs))


var_agent_configs = OrderedDict({

    "lr": "float, log, [1e-5, 1e-3]", # 1e-4
    "learning_starts": "int, [1, 2000]", # 500
    "target_network_update_freq": "int, log, [1, 2000]", # 800,
    "exploration_fraction": "float, [0.01, 0.99]", # 0.1,
    "n_step": "int, [1, 8]", # 1
    "buffer_size": "int, log, [33, 20000]", # ?? 1000000, # Sizes up to 32 crashed with Ray 0.7.3 (but not always!), size 16 did not crash with Ray 0.9.0dev
    "adam_epsilon": "float, log, [1e-12, 1e-1]", # ?? 1e-4,
    "train_batch_size": "cat, [4, 8, 16, 32, 64, 128]", # 32,

})

var_agent_configs = OrderedDict(sorted(var_agent_configs.items(), key=lambda t: t[0])) #hack because saved configs used below as random_configs are ordered alphabetically.

random_configs = \
[(1.86e-12, 1480, 0.0697, 311, 0.000545, 8, 1845, 64), # top 10 configs begin from here
 (1.13e-09, 842, 0.0503, 703, 0.00085, 4, 1114, 64),
 (2.2899999999999998e-05, 4711, 0.0665, 912, 0.000604, 6, 163, 16),
 (1.75e-11, 811, 0.0269, 1089, 0.0005780000000000001, 1, 9, 64),
 (1.37e-10, 3293, 0.0501, 1297, 0.000547, 7, 1, 64),
 (3.9e-06, 893, 0.0972, 655, 0.000657, 4, 1241, 4),
 (4e-09, 224, 0.01, 354, 0.00042, 4, 1776, 64),
 (0.000407, 6480, 0.0188, 1235, 0.000851, 8, 441, 32),
 (9.359999999999999e-07, 605, 0.0608, 706, 0.000299, 1, 250, 16),
 (1.99e-06, 17128, 0.0979, 1116, 0.00014, 5, 10, 64),
 (0.0524, 37, 0.965, 18, 1.49e-05, 1, 2, 64), # bottom 10 configs from here
 (0.0858, 6968, 0.581, 714, 1.86e-05, 7, 4, 128),
 (0.0966, 84, 0.9079999999999999, 1100, 1.93e-05, 1, 15, 4),
 (0.0524, 4394, 0.129, 1174, 2.9600000000000005e-05, 8, 16, 32),
 (0.0452, 33, 0.33, 1723, 1.09e-05, 3, 1, 64),
 (0.078, 84, 0.107, 436, 2.09e-05, 2, 5, 128),
 (0.0655, 156, 0.089, 437, 2.7899999999999997e-05, 5, 1521, 16),
 (0.0292, 2260, 0.34, 1907, 1.76e-05, 8, 2, 32),
 (0.0133, 6541, 0.218, 1393, 1.21e-05, 1, 3, 16),
 (0.0515, 507, 0.48100000000000004, 1866, 1.23e-05, 3, 136, 128)]

for i in range(len(random_configs)):
    random_configs[i] = tuple(random_configs[i]) ##IMP I think these are tuples because cartesian_product_configs by default has tuples.

var_configs = OrderedDict({
"env": var_env_configs,
"agent": var_agent_configs,

})

env_config = {
    "env": "GymEnvWrapper-Atari",
    "env_config": {
        "AtariEnv": {
            "game": 'qbert',
            'obs_type': 'image',
            'frameskip': 1,
        },
        # "GymEnvWrapper": {
        "atari_preprocessing": True,
        'frame_skip': 4,
        'grayscale_obs': False,
        'state_space_type': 'discrete',
        'action_space_type': 'discrete',
        'seed': 0,
        # },
        # 'seed': 0, #seed
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
    "schedule_max_timesteps": 10_000_000,
    # "learning_starts": 500,
    # "target_network_update_freq": 800,
    # "n_step": 4,
    "noisy": False,
    "num_atoms": 10, # [5, 10, 20]
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.75, #
    "prioritized_replay_beta": 0.4,
    "final_prioritized_replay_beta": 1.0, #
    "beta_annealing_fraction": 1.0, #
    # "hiddens": None,
    'hiddens': [512],

    "sample_batch_size": 4,
    "timesteps_per_iteration": 10000,
    # "train_batch_size": 32,
    "min_iter_time_s": 0,

    'num_gpus': 0,
    "num_workers": 3, # extra workers I think
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
    # "model": {
    #     "fcnet_hiddens": [256, 256],
    #     "fcnet_activation": "tanh",
    #     "use_lstm": False,
    #     "max_seq_len": 20,
    #     "lstm_cell_size": 256,
    #     "lstm_use_prev_action_reward": False,
    # },
}

from ray import tune
eval_config = {
    "evaluation_interval": None, # I think this means every x training_iterations
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
