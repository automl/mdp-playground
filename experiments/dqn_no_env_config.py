num_seeds = 10
from collections import OrderedDict
var_env_configs = OrderedDict({
})

var_configs = OrderedDict({
"env": var_env_configs
})

env_config = {
    "env": "RLToy-v0",
    "horizon": 100,
    "env_config": {
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
    "lr": 1e-4, # "lr": grid_search([1e-2, 1e-4, 1e-6]),
    "n_step": 1,
    "noisy": False,
    "num_atoms": 1,
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.5,
    "sample_batch_size": 4,
    "schedule_max_timesteps": 20000,
    "target_network_update_freq": 800,
    "timesteps_per_iteration": 1000,
    "min_iter_time_s": 0,
    "train_batch_size": 32,
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
    "evaluation_interval": 1, # I think this means every x training_iterations
    "evaluation_config": {
        "explore": False,
        "exploration_fraction": 0,
        "exploration_final_eps": 0,
        "evaluation_num_episodes": 10,
        "horizon": 100,
        "env_config": {
        }
    },
}
