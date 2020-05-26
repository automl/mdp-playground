'''###IMP dummy_seed should always be last in the order in the OrderedDict below!!!
'''
num_seeds = 10

import itertools
transforms = ['shift', 'scale', 'flip', 'rotate']
image_transforms = []
for i in range(len(transforms) + 1):
    curr_combos = list(itertools.combinations(transforms, i))
    for j in range(len(curr_combos)):
        if i == 0:
            curr_elem = 'none' # this is written to a CSV file with ' ' separater, therefore it needs to have some value in there.
        else:
            curr_elem = ''
        for k in range(i):
            curr_elem += curr_combos[j][k] + ','
        # print(curr_elem, i, j)
        image_transforms.append(curr_elem)

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
    'image_transforms': ['none', 'shift', 'scale', 'flip', 'rotate', 'shift,scale,rotate,flip'], # image_transforms,
    'image_scale_range': [(0.5, 2)],
    'image_width': [100],
    'image_height': [100],
    'dummy_seed': [i for i in range(num_seeds)],
})

var_configs = OrderedDict({
"env": var_env_configs
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
    "beta_annealing_fraction": 1.0,
    "buffer_size": 1000000,
    "double_q": False,
    "dueling": False,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.1,
    "final_prioritized_replay_beta": 1.0,
    "hiddens": None,
    "learning_starts": 1000,
    "lr": 1e-5, # "lr": grid_search([1e-2, 1e-4, 1e-6]),
    "n_step": 1,
    "noisy": False,
    "num_atoms": 1,
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.5,
    "sample_batch_size": 4, # Renamed from sample_batch_size in some Ray version
    "schedule_max_timesteps": 20000,
    "target_network_update_freq": 800,
    "timesteps_per_iteration": 1000,
    "min_iter_time_s": 0,
    "train_batch_size": 32,
}


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

model_config = {
    "model": {
        "fcnet_hiddens": [256, 256],
        # "custom_preprocessor": "ohe",
        "custom_options": {},  # extra options to pass to your preprocessor
        "conv_activation": "relu",
        "conv_filters": filters_100x100,
        # "no_final_linear": False,
        # "vf_share_layers": True,
        # "fcnet_activation": "tanh",
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action_reward": False,
    },
}
