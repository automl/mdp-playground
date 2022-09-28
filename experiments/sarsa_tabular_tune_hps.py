num_seeds = 10
from collections import OrderedDict
var_env_configs = OrderedDict({
    'state_space_size': [8],#, 10, 12, 14] # [2**i for i in range(1,6)]
    'action_space_size': [8],#2, 4, 8, 16] # [2**i for i in range(1,6)]
    'delay': [0], # + [2**i for i in range(4)],
    'sequence_length': [1],#i for i in range(1,4)]
    'reward_density': [0.25], # np.linspace(0.0, 1.0, num=5)
    'make_denser': [False],
    'terminal_state_density': [0.25], # np.linspace(0.1, 1.0, num=5)
    'transition_noise': [0],#, 0.01, 0.02, 0.10, 0.25]
    'reward_noise': [0],#, 1, 5, 10, 25] # Std dev. of normal dist.
    'dummy_seed': [i for i in range(num_seeds)],
})


var_agent_configs = OrderedDict({
    # learning rate used in TD updates
    "alpha": [.1, .3, .5],
    # agent epsilon value. Used as start value when decay linear or log. Otherwise constant value.
    "epsilon": [1e-1, 1e-2, 1e-3],
    # agent epsilon decay schedule, in (linear, log, const)
    "epsilon_decay": ["linear", "log", "const"],
})

var_configs = OrderedDict({
"env": var_env_configs,
"agent": var_agent_configs
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

import yaml

with open("tabular_rl/sarsa_config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

env_name = config["env_name"]
agent_name = config["agent_name"]

agent_config = config["agents"][agent_name]

eval_eps = config["eval_eps"]
seed = config["seed"]
no_render = config["no_render"]
discount_factor = config["discount_factor"]
alpha = agent_config["alpha"]

episodes = agent_config["episodes"]
env_max_steps = agent_config["env_max_steps"]
agent_eps_decay = agent_config["agent_eps_decay"]
agent_eps = agent_config["agent_eps"]

agent_config = {
    #"env_max_steps": env_max_steps,
    "num_episodes": episodes,
    "epsilon_decay": agent_eps_decay,
    "epsilon": agent_eps,
    "render_eval": no_render,
    "discount_factor": discount_factor,
    "alpha": alpha,
    "eval_every": eval_eps,
    #"timesteps_per_iteration": timesteps_per_iteration, #todo: perhaps pass this later as an argument to the agent
}

algorithm = "sarsa_tabular_tune_hps"
# agent_config = {
#     "adam_epsilon": 1e-4,
#     "beta_annealing_fraction": 1.0,
#     "buffer_size": 1000000,
#     "double_q": False,
#     "dueling": False,
#     "exploration_final_eps": 0.01,
#     "exploration_fraction": 0.1,
#     "final_prioritized_replay_beta": 1.0,
#     "hiddens": None,
#     "learning_starts": 1000,
#     "lr": 1e-4, # "lr": grid_search([1e-2, 1e-4, 1e-6]),
#     "n_step": 1,
#     "noisy": False,
#     "num_atoms": 1,
#     "prioritized_replay": False,
#     "prioritized_replay_alpha": 0.5,
#     "sample_batch_size": 4,
#     "schedule_max_timesteps": 20000,
#     "target_network_update_freq": 800,
#     "timesteps_per_iteration": 1000,
#     "min_iter_time_s": 0,
#     "train_batch_size": 32,
# }


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
