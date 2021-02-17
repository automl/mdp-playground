# Configuration Files

Configuration files are an easy way to define the experiments to run in mdp-playground. They can be passed as command line arguments to run_experiments.py either by using `-c` or `--config-file` argument. The example below shows how to run the sequence length and delay experiment for dqn specifying the configuration file through the command line. This file must be a Python file.

    run_experiments.py -c ./experiments/dqn_seq_del.py
    run_experiments.py --config-file ./experiments/dqn_seq_del.py

 There are 2 types of configurations across the experiments:
 - Variable Config and
 - Static Config
 
 ## Variable configurations

Variable configurations allow you to define variables whose impact is desired to be studied. For instance one might be interested on the effect on the agents' performance when varying the `sequence_length` and `delay` meta-features for the current experiment. Then `delay`  and `sequence_length` would be a key in `var_env_configs` dict and its corresponding value would be a *list of values they can take*.  Then a cartesian product of these lists is taken to generate various possible configurations to be run. 

    var_env_configs = OrderedDict({
		'state_space_size': [8],
		'action_space_size': [8],
		'delay': [0] + [2**i for i in  range(4)],
		'sequence_length': [1, 2, 3, 4],
		'reward_density': [0.25],
		'make_denser': [False],
		'terminal_state_density': [0.25],
		'transition_noise': [0],
		'reward_noise': [0],
		'dummy_seed': [i for i in  range(num_seeds)]
	})
	
    var_configs = OrderedDict({
    	    "env": var_env_configs,
     })

Variable configurations can be specified for either the environment, agent or the model across the current experiment. This can be specified through the OrderedDicts `var_env_configs`, `var_agent_configs` and `var_model_configs`  configuration options respectively.

Because Ray does not have a common way to address this specification of configurations for its agents, we offer the utility set `var_agent_configs` and `var_model_configs` the same way as `var_env_configs` is specified.

    var_configs = OrderedDict({
    	    "env": var_env_configs,
    	    "agent": var_agent_configs,
    	    "model" : var_model_configs 
     })


 Please see sample experiment config files in the experiments directory to see how to set the values for a given algorithm. 

## Static Configurations
`env_config`, `agent_config` and `model_config` are dictionaries which hold the static configuration for the current experiment as a normal Python dict.

**Example**:

    env_config = {
    "env": "RLToy-v0",
    "horizon": 100,
    "env_config": {
	    'seed': 0,
	    'state_space_type': 'discrete',
	    'action_space_type': 'discrete',
	    'generate_random_mdp': True,
	    'repeats_in_sequences': False,
	    'reward_scale': 1.0,
	    'completely_connected': True,
	    },
    }
Please look into the code for details.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTIwMDgzNDAyMCw5ODA0NTQ0ODcsMTE0MD
A3ODc3NCwtMTQyODk4MjI4LC0xNzIzNDk4MDQwLDEwNTIzNjcx
OV19
-->

## API for agents
If you implement an agent and want to use our run_experiments.py out of the box then agent must follow the following API:
1. It should have a config that can be passed as dictionary
2. The configuration values of the **agent** should be specified as follows:

### Exploration
* **schedule_max_timesteps (int):**
Max num timesteps for annealing schedules. Exploration is annealed from 1.0 to exploration_fraction over this number of timesteps scaled by exploration_fraction

* **timesteps_per_iteration (int):**
Number of env steps to optimize for before returning

* **exploration_fraction (float):**
Fraction of entire training period over which the exploration rate is annealed

* **exploration_final_eps (float):**
Final value of random action probability

* **target_network_update_freq (int):**
Update the target network every `target_network_update_freq` steps.


### Replay buffer configuration
* **buffer_size (int):**
Size of the replay buffer.

* **prioritized_replay (bool):**
If True prioritized replay buffer will be used

* **prioritized_replay_alpha (float):** 
Alpha parameter for prioritized replay buffer

* **prioritized_replay_beta (float):** 
Beta parameter for sampling from prioritized replay buffer.

* **beta_annealing_fraction (float):** 
Fraction of entire training period over which the beta parameter is annealed.

* **final_prioritized_replay_beta (float):** 
Final value of beta

* **prioritized_replay_eps (float):** 
Epsilon to add to the TD errors when updating priorities.

### Optimization:
* **lr (float):** 
Learning rate for adam optimizer

* **adam_epsilon (float):** 
Adam epsilon hyper parameter

* **grad_norm_clipping (float):** 
If not None, clip gradients during optimization at this value.

* **learning_starts (int):** 
How many steps of the model to sample before learning starts.

* **train_batch_size (int):** 
Size of a batched sampled from replay buffer for training. Note that if async_updates is set, then each worker returns gradients for a batch of this size.

### Execution:   
* **num_envs_per_worker (int):** 
Number of environments to evaluate vectorwise per worker.

* **sample_batch_size (int):** 
Default sample batch size (unroll length). Batches of this size are collected from workers until train_batch_size is met. When using multiple envs per worker, this is multiplied by num_envs_per_worker. The replay buffer is updated with this many samples at once. Note that this setting applies per-worker if num_workers > 1.

* **train_batch_size (int):** 
Size of a batched sampled from replay buffer for training. Note that if async_updates is set, then each worker returns gradients for a batch of this size.
Should be >= sample_batch_size. Samples batches will be concatenated together to this size for training.
 
 * **batch_mode (str):** 
Whether to rollout "complete_episodes" or "truncate_episodes".

 * **sample_async (bool):** 
Use a background thread for sampling (slightly off-policy)

### DQN specifics:
* **noisy (bool):** 
Whether to use noisy network

* **dueling (bool):** 
Whether to use dueling dqn

* **double_q (bool):** 
Whether to use double dqn

* **hiddens ( [int] vector):** 
Postprocess model outputs with these hidden layers to compute the state and action values.

* **n_step (int):** 
N-step Q learning
 