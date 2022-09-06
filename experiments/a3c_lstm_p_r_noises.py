import itertools
from ray import tune
from collections import OrderedDict

num_seeds = 10

var_env_configs = OrderedDict(
    {
        "state_space_size": [8],  # , 10, 12, 14] # [2**i for i in range(1,6)]
        "action_space_size": [8],  # 2, 4, 8, 16] # [2**i for i in range(1,6)]
        "delay": [0],
        "sequence_length": [1],  # i for i in range(1,4)]
        "reward_density": [0.25],  # np.linspace(0.0, 1.0, num=5)
        "make_denser": [False],
        "terminal_state_density": [0.25],  # np.linspace(0.1, 1.0, num=5)
        "transition_noise": [0, 0.01, 0.02, 0.10, 0.25],
        "reward_noise": [0, 1, 5, 10, 25],  # Std dev. of normal dist.
        "dummy_seed": [i for i in range(num_seeds)],
    }
)

var_configs = OrderedDict({"env": var_env_configs})

env_config = {
    "env": "RLToy-v0",
    "horizon": 100,
    "env_config": {
        "seed": 0,  # seed
        "state_space_type": "discrete",
        "action_space_type": "discrete",
        "generate_random_mdp": True,
        "repeats_in_sequences": False,
        "reward_scale": 1.0,
        "completely_connected": True,
    },
}

algorithm = "A3C"
agent_config = {
    # Size of rollout batch
    "sample_batch_size": 10,  # maybe num_workers * sample_batch_size * num_envs_per_worker * grads_per_step
    "train_batch_size": 100,  # seems to have no effect
    # Use PyTorch as backend - no LSTM support
    "use_pytorch": False,
    # GAE(gamma) parameter
    "lambda": 0.0,  #
    # Max global norm for each gradient calculated by worker
    "grad_clip": 10.0,  # low prio.
    # Learning rate
    "lr": 0.0001,  #
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.1,  #
    # Entropy coefficient
    "entropy_coeff": 0.1,  #
    # Min time per iteration
    "min_iter_time_s": 0,
    # Workers sample async. Note that this increases the effective
    # sample_batch_size by up to 5x due to async buffering of batches.
    "sample_async": True,
    "timesteps_per_iteration": 1000,
    "num_workers": 3,
    "num_envs_per_worker": 5,
    "optimizer": {"grads_per_step": 10},
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
value_tuples = []
for config_type, config_dict in var_configs.items():
    for key in config_dict:
        assert isinstance(
            var_configs[config_type][key], list
        ), "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
        value_tuples.append(var_configs[config_type][key])


cartesian_product_configs = list(itertools.product(*value_tuples))
print("Total number of configs. to run:", len(cartesian_product_configs))
