'''###IMP dummy_seed should always be last in the order in the OrderedDict below!!!
'''
num_seeds = 10

from collections import OrderedDict
var_env_configs = OrderedDict({
    'state_space_dim': [2],
    'action_space_dim': [2],
    'delay': [0],
    # 'sequence_length': [1], #, 2, 3, 4],#i for i in range(1,4)]
    # 'reward_density': [0.25], # np.linspace(0.0, 1.0, num=5)
    'make_denser': [True],
    # 'terminal_state_density': [0.25], # np.linspace(0.1, 1.0, num=5)
    'transition_noise': [0],
    'reward_noise': [0],
    'target_point': [[0, 0]],
    "target_radius": [0.5],
    "state_space_max": [10],
    "action_space_max": [1],
    "action_loss_weight": [0.01, 0.1, 1, 5, 10, 100],
    'time_unit': [1.0],
    'transition_dynamics_order': [1],
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
        'state_space_type': 'continuous',
        'action_space_type': 'continuous',
        'inertia': 1,
        'reward_scale': 1.0,
        "reward_function": 'move_to_a_point',
        # 'make_denser': True,
        # "log_level": 'INFO',
        "log_filename": '/tmp/sac_mv_pt.log',
    },
}

algorithm = "SAC"
agent_config = {
    # Initial value to use for the entropy weight alpha.
    "initial_alpha": 0.1,
    "optimization": {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.0002,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 2000,

    # Apply a state preprocessor with spec given by the "model" config option
    # (like other RL algorithms). This is mostly useful if you have a weird
    # observation shape, like an image. Disabled by default.
    "use_state_preprocessor": False,
    # N-step Q learning
    "n_step": 1,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,

    "buffer_size": 50000,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Time steps over which the beta parameter is annealed.
    # "prioritized_replay_beta_annealing_timesteps": 20000,
    # Final value of beta
    "final_prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,

    # "schedule_max_timesteps": 20000,
    "timesteps_per_iteration": 1000,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    # "rollout_fragment_length": 1,
    "rollout_fragment_length": 1, # Renamed from sample_batch_size in some Ray version
    "train_batch_size": 32,
    "min_iter_time_s": 0,
}


model_config = {
    "Q_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [32, 32],
    },
    "policy_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [32, 32],
    },

    "model": {
        # "custom_preprocessor": "ohe",
        "custom_options": {},  # extra options to pass to your preprocessor
        # "no_final_linear": False,
        # "vf_share_layers": True,
        # "fcnet_activation": "tanh",
        "use_lstm": False,
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
            "dummy_eval": True, #hack Used to check if we are in evaluation mode or training mode inside Ray callback on_episode_end() to be able to write eval stats
            'transition_noise': 0 if "state_space_type" in env_config["env_config"] and env_config["env_config"]["state_space_type"] == "discrete" else tune.function(lambda a: a.normal(0, 0)),
            'reward_noise': tune.function(lambda a: a.normal(0, 0)),
            'action_loss_weight': 0.0,
        }
    },
}
