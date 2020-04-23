num_seeds = 10
state_space_sizes = [8]#, 10, 12, 14] # [2**i for i in range(1,6)]
action_space_sizes = [8]#2, 4, 8, 16] # [2**i for i in range(1,6)]
delays = [0] + [2**i for i in range(4)]
sequence_lengths = [1, 2, 3, 4]#i for i in range(1,4)]
reward_densities = [0.25] # np.linspace(0.0, 1.0, num=5)
make_densers = [False]
# make_reward_dense = [True, False]
terminal_state_densities = [0.25] # np.linspace(0.1, 1.0, num=5)
transition_noises = [0]#, 0.01, 0.02, 0.10, 0.25]
reward_noises = [0]#, 1, 5, 10, 25] # Std dev. of normal dist.
algorithms = ["DQN"]
seeds = [i for i in range(num_seeds)]
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
