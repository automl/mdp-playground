'''###IMP dummy_seed should always be last in the order in the OrderedDict below!!!
'''
num_seeds = 3

from collections import OrderedDict
var_env_configs = OrderedDict({
    'state_space_size': [8],#, 10, 12, 14] # [2**i for i in range(1,6)]
    'action_space_size': [8],#2, 4, 8, 16] # [2**i for i in range(1,6)]
    'delay': [0], # + [2**i for i in range(4)],
    'sequence_length': [1], #, 2, 3, 4],#i for i in range(1,4)]
    'reward_density': [0.25], # np.linspace(0.0, 1.0, num=5)
    'make_denser': [False],
    'terminal_state_density': [0.25], # np.linspace(0.1, 1.0, num=5)
    'transition_noise': [0],#, 0.01, 0.02, 0.10, 0.25]
    'reward_noise': [0],#, 1, 5, 10, 25] # Std dev. of normal dist.
    'image_representations': [True],
    'image_transforms': ['none'], #image_transforms, # ['shift', 'scale', 'flip', 'rotate', 'shift,scale,rotate,flip']
    'image_width': [100],
    'image_height': [100],
    'dummy_seed': [i for i in range(num_seeds)],
})

var_agent_configs = OrderedDict({
    "learning_starts": [500, 1000, 2000],
    "lr": [1e-3, 1e-4, 1e-5], # "lr": grid_search([1e-2, 1e-4, 1e-6]),
    "n_step": [1],
    "target_network_update_freq": [8, 80, 800],
})


# formula [(W−K+2P)/S]+1; for padding=same: P = ((S-1)*W - S + K)/2
filters_84x84 = [
    [16, [8, 8], 4], # changes from 84x84x1 with padding 4 to 22x22x16 (or 26x26x16 for 100x100x1)
    [32, [4, 4], 2], # changes to 11x11x32 with padding 2 (or 13x13x32 for 100x100x1)
    [256, [11, 11], 1], # changes to 1x1x256 with padding 0 (or 3x3x256 for 100x100x1); this is the only layer with valid padding in Ray!
]

filters_100x100 = [
    [16, [8, 8], 4], # changes from 84x84x1 with padding 4 to 22x22x16 (or 26x26x16 for 100x100x1)
    [32, [4, 4], 2], # changes to 11x11x32 with padding 2 (or 13x13x32 for 100x100x1)
    [64, [13, 13], 1], # changes to 1x1x64 with padding 0 (or 3x3x64 for 100x100x1); this is the only layer with valid padding in Ray!
]
# [num_outputs(=8 in this case), [1, 1], 1] conv2d appended by Ray always followed by a Dense layer with 1 output

# filters_99x99 = [
#     [16, [8, 8], 4], # 51x51x16
#     [32, [4, 4], 2],
#     [64, [13, 13], 1],
# ]

filters_100x100_large = [
    [16, [8, 8], 4],
    [32, [4, 4], 2],
    [256, [13, 13], 1],
]

filters_50x50 = [
    [16, [4, 4], 2],
    [32, [4, 4], 2],
    [64, [13, 13], 1],
]

filters_400x400 = [
    [16, [32, 32], 16],
    [32, [4, 4], 2],
    [64, [13, 13], 1],
]


var_model_configs = OrderedDict({
    "conv_filters": [filters_100x100, filters_100x100_large],
})

var_configs = OrderedDict({
"env": var_env_configs,
"agent": var_agent_configs,
"model": var_model_configs,
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
    "adam_epsilon": 1e-4,
    "buffer_size": 1000000,
    "double_q": True,
    "dueling": True,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.1,
    "schedule_max_timesteps": 20000,
    # "hiddens": None,
    "noisy": True,
    "num_atoms": 10, # [5, 10, 20]
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.75, #
    "prioritized_replay_beta": 0.4,
    "final_prioritized_replay_beta": 1.0, #
    "beta_annealing_fraction": 1.0, #

    "sample_batch_size": 4,
    "timesteps_per_iteration": 1000,
    "train_batch_size": 32,
    "min_iter_time_s": 0,
}


model_config = {
    "model": {
        "fcnet_hiddens": [256, 256],
        # "custom_preprocessor": "ohe",
        "custom_options": {},  # extra options to pass to your preprocessor
        "conv_activation": "relu",
        # "no_final_linear": False,
        # "vf_share_layers": True,
        # "fcnet_activation": "tanh",
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action_reward": False,
    },
}
