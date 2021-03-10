'''###IMP dummy_seed should always be last in the order in the OrderedDict below!!!
'''
num_seeds = 10

from collections import OrderedDict
var_env_configs = OrderedDict({
    'dummy_seed': [i for i in range(num_seeds)],
})

var_agent_configs = OrderedDict({
    "policy_delay": [2, 3, 4],
    # N-step Q learning
    "n_step": [2, 3, 4],
    "target_noise": [0.02, 0.2, 2.0],
    "target_noise_clip": [1.5, 2.5], # this is abs. val. for our purposes we need relative so we hack that in run_experiments.py
})

# var_model_configs = OrderedDict({
# })

var_configs = OrderedDict({
"env": var_env_configs,
"agent": var_agent_configs,
# "model": var_model_configs,
})

env_config = {
    "env": "PusherWrapper-v2",
    "horizon": 100,
    "env_config": {
        'state_space_type': 'continuous',
        'action_space_type': 'continuous',
        "MujocoEnv": {
        },
    },
}

algorithm = "TD3"
agent_config = {
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 1e-4,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-4,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.02,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 5000,

    "critic_hiddens": [128, 128],
    "actor_hiddens": [128, 128],

    "twin_q": True,
    "smooth_target_policy": True,

    # Update the target network every `target_network_update_freq` steps.
#    "target_network_update_freq": 0,

    "buffer_size": 1000000,

    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,

    # "schedule_max_timesteps": 20000,
    "timesteps_per_iteration": 1000,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    # "rollout_fragment_length": 1,
    "rollout_fragment_length": 1, # Renamed from sample_batch_size in some Ray version
    "train_batch_size": 256,
    "min_iter_time_s": 0,
    "num_workers": 0,
    "num_gpus": 0,
#    "evaluation_interval": 1,

}


model_config = {
}

from ray import tune
eval_config = {
    "evaluation_interval": 1, # I think this means every x training_iterations
    "evaluation_config": {
        "explore": False,
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
value_tuples = []
for config_type, config_dict in var_configs.items():
    for key in config_dict:
        assert type(var_configs[config_type][key]) == list, "var_config should be a dict of dicts with lists as the leaf values to allow each configuration option to take multiple possible values"
        value_tuples.append(var_configs[config_type][key])

import itertools
cartesian_product_configs = list(itertools.product(*value_tuples))
print("Total number of configs. to run:", len(cartesian_product_configs))
