"""###IMP dummy_seed should always be last in the order in the OrderedDict below!!!
"""
import itertools
from ray import tune
from collections import OrderedDict
num_seeds = 10


var_env_configs = OrderedDict(
    {
        "action_space_max": [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        "dummy_seed": [i for i in range(num_seeds)],
    }
)

var_configs = OrderedDict({"env": var_env_configs})

env_config = {
    "env": "ReacherWrapper-v2",
    "horizon": 100,
    "soft_horizon": False,
    "env_config": {
        "state_space_type": "continuous",
        "action_space_type": "continuous",
        "MujocoEnv": {},
    },
}

algorithm = "DDPG"
agent_config = {
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 1e-3,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-3,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.002,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 10000,
    "critic_hiddens": [256, 256],
    "actor_hiddens": [256, 256],
    # N-step Q learning
    "n_step": 1,
    # Update the target network every `target_network_update_freq` steps.
    # "target_network_update_freq": 1,
    "buffer_size": 1000000,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,
    # "schedule_max_timesteps": 20000,
    "timesteps_per_iteration": 1000,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    # "rollout_fragment_length": 1,
    "rollout_fragment_length": 1,  # Renamed from sample_batch_size in some Ray version
    "train_batch_size": 256,
    "min_iter_time_s": 0,
    "num_workers": 0,
    "num_gpus": 0,
    #    "evaluation_interval": 1,
}


model_config = {}


eval_config = {
    "evaluation_interval": 1,  # I think this means every x training_iterations
    "evaluation_config": {
        "explore": False,
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
        assert (
            isinstance(var_configs[config_type][key], list)
        ), "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
        value_tuples.append(var_configs[config_type][key])


cartesian_product_configs = list(itertools.product(*value_tuples))
print("Total number of configs. to run:", len(cartesian_product_configs))
