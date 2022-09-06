from ray import tune
from collections import OrderedDict

num_seeds = 1
timesteps_total = 20_000
num_configs = 1000


sobol_env_configs = OrderedDict(
    {
        "action_space_size": (8,),  # , 10, 12, 14] # [2**i for i in range(1,6)]
        # 'action_space_size': (64),#2, 4, 8, 16] # [2**i for i in range(1,6)]
        "delay": "cat, " + str([i for i in range(11)]),  # + [2**i for i in range(4)],
        "sequence_length": "cat, "
        + str([i for i in range(1, 4)]),  # , 2, 3, 4],#i for i in range(1,4)]
        "diameter": "cat, "
        + str([2 ** i for i in range(4)]),  # + [2**i for i in range(4)],
        "reward_density": "float, [0.03, 0.5]",  # np.linspace(0.0, 1.0, num=5)
        "make_denser": (False,),
        "terminal_state_density": (0.25,),  # np.linspace(0.1, 1.0, num=5)
        "reward_dist": "float, [0.01, 0.8]",
        "reward_scale": "float, log, [0.1, 100]",
        "dummy_seed": (0,),  # "cat, " + str([i for i in range(num_seeds)]), #seed
    }
)


print(sobol_env_configs)

random_agent_configs = OrderedDict(
    {
        "lr": "float, log, [1e-5, 1e-3]",  # 1e-4
        "learning_starts": "int, [1, 2000]",  # 500
        "target_network_update_freq": "int, log, [1, 2000]",  # 800,
        "exploration_fraction": "float, [0.01, 0.99]",  # 0.1,
        "n_step": "int, [1, 8]",  # 1
        # ?? 1000000, # Sizes up to 32 crashed with Ray 0.7.3 (but not always!), size 16 did not crash with Ray 0.9.0dev
        "buffer_size": "int, log, [33, 20000]",
        "adam_epsilon": "float, log, [1e-12, 1e-1]",  # ?? 1e-4,
        "train_batch_size": "cat, [4, 8, 16, 32, 64, 128]",  # 32,
    }
)

# hack because ConfigSpace below orders alphabetically, the returned
# configs are in a jumbled order compared to the order above, which would
# create problems with config processing.
random_agent_configs = OrderedDict(
    sorted(random_agent_configs.items(), key=lambda t: t[0])
)


random_configs = OrderedDict(
    {
        "env": {},
        "agent": random_agent_configs,
    }
)


sobol_configs = OrderedDict(
    {
        "env": sobol_env_configs,
    }
)


env_config = {
    "env": "RLToy-v0",
    "horizon": 100,
    "env_config": {
        "seed": 0,  # seed
        "state_space_type": "discrete",
        "action_space_type": "discrete",
        "generate_random_mdp": True,
        "repeats_in_sequences": False,
        # 'reward_scale': 1.0,
        "completely_connected": True,
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
    "noisy": False,
    "num_atoms": 10,  # [5, 10, 20]
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.75,  #
    "prioritized_replay_beta": 0.4,
    "final_prioritized_replay_beta": 1.0,  #
    "beta_annealing_fraction": 1.0,  #
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
        "device_count": {"CPU": 2},
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
        # "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action_reward": False,
    },
}


eval_config = {
    "evaluation_interval": 1,  # I think this means every x training_iterations
    "evaluation_config": {
        "explore": False,
        "exploration_fraction": 0,
        "exploration_final_eps": 0,
        "evaluation_num_episodes": 10,
        "horizon": 100,
        "env_config": {
            "dummy_eval": True,  # hack Used to check if we are in evaluation mode or training mode inside Ray callback on_episode_end() to be able to write eval stats
            "transition_noise": 0
            if "state_space_type" in env_config["env_config"]
            and env_config["env_config"]["state_space_type"] == "discrete"
            else tune.function(lambda a: a.normal(0, 0)),
            "reward_noise": tune.function(lambda a: a.normal(0, 0)),
            "action_loss_weight": 0.0,
        },
    },
}
