num_seeds = 5
from collections import OrderedDict
var_env_configs = OrderedDict({
    'reward_noise': [0, 1, 5, 10, 25], # Std dev. of normal dist.
    'dummy_seed': [i for i in range(num_seeds)],
})

var_configs = OrderedDict({
"env": var_env_configs
})

env_config = {
    "env": "GymEnvWrapper-v0",
    "env_config": {
        "AtariEnv": {
            "game": "breakout",
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
agent_config = { # Taken from Ray tuned_examples
    'adam_epsilon': 0.00015,
    'buffer_size': 1000000,
    'double_q': False,
    'dueling': False,
    'exploration_config': {   'epsilon_timesteps': 200000,
                           'final_epsilon': 0.01},
    'final_prioritized_replay_beta': 1.0,
    'hiddens': [512],
    'learning_starts': 20000,
    'lr': 6.25e-05,
    'n_step': 1,
    'noisy': False,
    'num_atoms': 1,
    'num_gpus': 0,
    'num_workers': 10,
    'prioritized_replay': False,
    'prioritized_replay_alpha': 0.5,
    'prioritized_replay_beta_annealing_timesteps': 2000000,
    'rollout_fragment_length': 4,
    'target_network_update_freq': 8000,
    'timesteps_per_iteration': 10000,
    'train_batch_size': 32,
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
    "evaluation_interval": 10, # I think this means every x training_iterations
    "evaluation_config": {
        "explore": False,
        "exploration_fraction": 0,
        "exploration_final_eps": 0,
        "evaluation_num_episodes": 10,
        # "horizon": 100,
        "env_config": {
            "dummy_eval": True, #hack Used to check if we are in evaluation mode or training mode inside Ray callback on_episode_end() to be able to write eval stats
            'transition_noise': 0 if "state_space_type" in env_config["env_config"] and env_config["env_config"]["state_space_type"] == "discrete" else tune.function(lambda a: a.normal(0, 0)),
            'reward_noise': tune.function(lambda a: a.normal(0, 0)),
            'action_loss_weight': 0.0,
        }
    },
}
