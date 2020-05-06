num_seeds = 10
from collections import OrderedDict
env_configs = OrderedDict({
    'state_space_size': [8],
    'action_space_size': [8],
    'delay': [0],
    'sequence_length': [1],
    'reward_density': [0.25, 0.5, 0.75],
    'make_denser': [False],
    'terminal_state_density': [0.25],
    'transition_noise': [0],
    'reward_noise': [0],
    'dummy_seed': [i for i in range(num_seeds)],
})

algorithm = "DQN"
agent_config = {
    "adam_epsilon": 1e-4,
    "buffer_size": 1000000,
    "double_q": True,
    "dueling": True,
    "lr": 1e-3,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.1,
    "schedule_max_timesteps": 20000,
    # "hiddens": None,
    "learning_starts": 500,
    "target_network_update_freq": 80,
    "n_step": 4, # delay + sequence_length [1, 2, 4, 8]
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
    "min_iter_time_s": 1,
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
