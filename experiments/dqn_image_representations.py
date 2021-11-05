"""
This files specifies different configurations to be run for an MDP Playground
experiment. The configurations are divided into varying and static configs.
The varying configs are the ones that will vary across this experiment. The
static configs remain fixed throughout an experiment. Additionally, evaluation
configurations are run interleaved with the experiment to evaluate the agent's
learning progress.

Varying configs can be specified for the environment, agent and the NN model used.
This is done as follows:

Specify var_configs as a dict of dicts with fixed keys:
"env" for holding configs to vary in the environment
"agent" for holding configs to vary for the agent
"model" for holding configs to vary for the NN used

Static configs are specified using:
env_config specifies static environment configurations
agent_config specifies static agent configurations
model_config specifies static NN model configurations
eval_config specifies static evaluation configurations

NOTE: Please note that for any configuration values not provided here, reasonable
default values would be used. As such, these config values are much more verbose
than needed. We only explicitly provide many important configuration values here
to have them be easy to find.
"""
from ray import tune
from collections import OrderedDict
import itertools
num_seeds = 10
timesteps_total = 20_000

# var_env_configs specifies variable configs in the environment and we use it as
# the value for the key "env" in var_configs:
var_env_configs = OrderedDict(
    {
        "state_space_size": [8],
        "action_space_size": [8],
        "delay": [0],
        "sequence_length": [1],
        "reward_density": [0.25],
        "make_denser": [False],
        "terminal_state_density": [0.25],
        "transition_noise": [0],
        "reward_noise": [0],
        "image_representations": [True],
        "image_transforms": [
            "none",
            "shift",
            "scale",
            "flip",
            "rotate",
            "shift,scale,rotate,flip",
        ],
        "image_scale_range": [(0.5, 2)],
        "image_width": [100],
        "image_height": [100],
        "dummy_seed": [i for i in range(num_seeds)],
    }
)

var_configs = OrderedDict({"env": var_env_configs})

# All the configs from here on are static configs, i.e., those that won't be
# varied in any runs in this experiment:
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
    "lr": 1e-5,
    "n_step": 1,
    "noisy": False,
    "num_atoms": 1,
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.5,
    "sample_batch_size": 4,  # Renamed from sample_batch_size in some Ray version
    "schedule_max_timesteps": 20000,
    "target_network_update_freq": 800,
    "timesteps_per_iteration": 1000,
    "min_iter_time_s": 0,
    "train_batch_size": 32,
}


# formula [(W−K+2P)/S]+1; for padding=same: P = ((S-1)*W - S + K)/2
filters_100x100 = [
    [
        16,
        [8, 8],
        4,
    ],  # changes from 84x84x1 with padding 4 to 22x22x16 (or 26x26x16 for 100x100x1)
    [32, [4, 4], 2],  # changes to 11x11x32 with padding 2 (or 13x13x32 for 100x100x1)
    [
        64,
        [13, 13],
        1,
    ],  # changes to 1x1x64 with padding 0 (or 3x3x64 for 100x100x1); this is the only layer with valid padding in Ray!
]

model_config = {
    "model": {
        "fcnet_hiddens": [256, 256],
        "custom_options": {},
        "conv_activation": "relu",
        "conv_filters": filters_100x100,
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action_reward": False,
    },
}


eval_config = {
    "evaluation_interval": 1,  #  this means every x training_iterations
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
