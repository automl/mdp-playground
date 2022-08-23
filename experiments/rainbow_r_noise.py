num_seeds = 100
from collections import OrderedDict
var_env_configs = OrderedDict({
    'state_space_size': [8],#, 10, 12, 14] # [2**i for i in range(1,6)]
    'action_space_size': [8],#2, 4, 8, 16] # [2**i for i in range(1,6)]
    'delay': [0], # + [2**i for i in range(4)],
    'sequence_length': [1],
    'reward_density': [0.25], # np.linspace(0.0, 1.0, num=5)
    'make_denser': [False],
    'terminal_state_density': [0.25], # np.linspace(0.1, 1.0, num=5)
    'transition_noise': [0], #, 0.01, 0.02, 0.10, 0.25],
    'reward_noise': [0, 1, 5, 10, 25], # Std dev. of normal dist.
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
    "min_iter_time_s": 0,
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
    "evaluation_interval": None,  # I think this means every x training_iterations
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

