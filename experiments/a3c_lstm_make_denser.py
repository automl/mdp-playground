num_seeds = 10
from collections import OrderedDict
env_configs = OrderedDict({
    'state_space_size': [8],#, 10, 12, 14] # [2**i for i in range(1,6)]
    'action_space_size': [8],#2, 4, 8, 16] # [2**i for i in range(1,6)]
    'delay': [0],
    'sequence_length': [2, 3, 4],#i for i in range(1,4)]
    'reward_density': [0.25], # np.linspace(0.0, 1.0, num=5)
    'make_denser': [True],
    'terminal_state_density': [0.25], # np.linspace(0.1, 1.0, num=5)
    'transition_noise': [0],#, 0.01, 0.02, 0.10, 0.25]
    'reward_noise': [0],#, 1, 5, 10, 25] # Std dev. of normal dist.
    'dummy_seed': [i for i in range(num_seeds)],
})

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
    "lr": 0.0001, #
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.1, #
    # Entropy coefficient
    "entropy_coeff": 0.1, #
    # Min time per iteration
    "min_iter_time_s": 0,
    # Workers sample async. Note that this increases the effective
    # sample_batch_size by up to 5x due to async buffering of batches.
    "sample_async": True,
    "timesteps_per_iteration": 1000,
    "num_workers": 3,
    "num_envs_per_worker": 5,

    "optimizer": {
        "grads_per_step": 10
    },
}

model_config = {
    "model": {
        "fcnet_hiddens": [128, 128, 128],
        "custom_preprocessor": "ohe",
        "custom_options": {},  # extra options to pass to your preprocessor
        "fcnet_activation": "tanh",
        "use_lstm": True,
        "lstm_cell_size": 64,
        "lstm_use_prev_action_reward": True,
    },
}
