num_seeds = 1
timesteps_total = 20_000
num_configs = 100

from collections import OrderedDict

random_agent_configs = OrderedDict({

    "lr": "float, log, [1e-5, 1e-3]", # 1e-4
    "learning_starts": "int, [10, 20000]", # 500
    "target_network_update_freq": "int, log, [10, 10000]", # 800,
    "exploration_fraction": "float, [0.01, 0.99]", # 0.1,
    "n_step": "int, [1, 16]", # 1
    "buffer_size": "int, log, [333, 500000]", # ?? 1000000, # Sizes up to 32 crashed with Ray 0.7.3 (but not always!), size 16 did not crash with Ray 0.9.0dev
    "adam_epsilon": "float, log, [1e-12, 1e-1]", # ?? 1e-4,
    "train_batch_size": "cat, [4, 8, 16, 32, 64, 128]", # 32,

})

random_agent_configs = OrderedDict(sorted(random_agent_configs.items(), key=lambda t: t[0])) #hack because ConfigSpace below orders alphabetically, the returned configs are in a jumbled order compared to the order above, which would create problems with config processing.


random_configs = OrderedDict({
"env": {},
"agent": random_agent_configs,

})


# These are currently needed to write dummy_seed to stats CSV. A seed column is
# needed for data loading

sobol_env_configs = OrderedDict({
    'dummy_seed': (0,), # "cat, " + str([i for i in range(num_seeds)]), #seed
})

# print(sobol_env_configs)

sobol_configs = OrderedDict({
"env": sobol_env_configs,

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
