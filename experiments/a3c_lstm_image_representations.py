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
            curr_elem = 'none' # this is written to a CSV file with ' ' separater, therefore it's good to have some value in there.
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

algorithm = "A3C"
agent_config = {
    # Size of rollout batch
    "sample_batch_size": 10, # maybe num_workers * sample_batch_size * num_envs_per_worker * grads_per_step
    "train_batch_size": 100, # seems to have no effect
    # Use PyTorch as backend - no LSTM support
    "use_pytorch": False,
    # GAE(gamma) parameter
    "lambda": 0.0, #
    # Max global norm for each gradient calculated by worker
    "grad_clip": 10.0, # low prio.
    # Learning rate
    "lr": 0.001, #
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5, #
    # Entropy coefficient
    "entropy_coeff": 0.1, #
    # Min time per iteration
    "min_iter_time_s": 0,
    # Workers sample async. Note that this increases the effective
    # sample_batch_size by up to 5x due to async buffering of batches.
    "sample_async": True,
    "timesteps_per_iteration": 7500,
    "num_workers": 3,
    "num_envs_per_worker": 5,

    "optimizer": {
        "grads_per_step": 10
    },
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
        "fcnet_hiddens": [128, 128, 128],
        # "custom_preprocessor": "ohe",
        "custom_options": {},  # extra options to pass to your preprocessor
        "conv_activation": "relu",
        "conv_filters": filters_100x100,
        # "no_final_linear": False,
        # "vf_share_layers": True,
        # "fcnet_activation": "tanh",
        "use_lstm": True,
        # "max_seq_len": 20,
        "lstm_cell_size": 64,
        "lstm_use_prev_action_reward": True,
    },
}
