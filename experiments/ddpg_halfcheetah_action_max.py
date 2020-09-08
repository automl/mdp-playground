'''###IMP dummy_seed should always be last in the order in the OrderedDict below!!!
'''
num_seeds = 5

from collections import OrderedDict
var_env_configs = OrderedDict({
    "action_space_max": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    'dummy_seed': [i for i in range(num_seeds)],
})

var_configs = OrderedDict({
"env": var_env_configs
})

env_config = {
    "env": "HalfCheetahWrapper-v3",
    "horizon": 1000,
    "soft_horizon": False,
    "env_config": {
    },
}

algorithm = "DDPG"
agent_config = {
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 3e-4,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 3e-4,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.005,
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
