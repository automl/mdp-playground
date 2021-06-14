from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import warnings
import logging
import copy
from datetime import datetime
import numpy as np
import scipy
from scipy import stats
from scipy.spatial import distance
import gym
from mdp_playground.spaces import (
    BoxExtended,
    DiscreteExtended,
    TupleExtended,
    ImageMultiDiscrete,
    ImageContinuous,
    GridActionSpace,
)


class RLToyEnv(gym.Env):
    """
    The base toy environment in MDP Playground. It is parameterised by a config dict and can be instantiated to be an MDP with any of the possible dimensions from the accompanying research paper. The class extends OpenAI Gym's environment gym.Env.

    The accompanying paper is available at: https://arxiv.org/abs/1909.07750.

    Instead of implementing a new class for every type of MDP, the intent is to capture as many common dimensions across different types of environments as possible and to be able to control the difficulty of an environment by allowing fine-grained control over each of these dimensions. The focus is to be as flexible as possible.

    The configuration for the environment is passed as a dict at initialisation and contains all the information needed to determine the dynamics of the MDP that the instantiated environment will emulate. We recommend looking at the examples in example.py to begin using the environment since the dimensions and config options are mostly self-explanatory. If you want to specify custom MDPs, please see the use_custom_mdp config option below. For more details, we list here the dimensions and config options (their names here correspond to the keys to be passed in the config dict):
        state_space_type : str
            Specifies what the environment type is. Options are "continuous", "discrete" and "grid". The "grid" environment is, basically, a discretised version of the continuous environment.
        delay : int >= 0
            Delays each reward by this number of timesteps.
        sequence_length : int >= 1
            Intrinsic sequence length of the reward function of an environment. For discrete environments, randomly selected sequences of this length are set to be rewardable at initialisation if use_custom_mdp = false and generate_random_mdp = true.
        transition_noise : float in range [0, 1] or Python function(rng)
            For discrete environments, it is a float that specifies the fraction of times the environment transitions to a noisy next state at each timestep, independently and uniformly at random.
            For continuous environments, if it's a float, it's used as the standard deviation of an i.i.d. normal distribution of noise. If it is a Python function with one argument, it is added to next state. The argument is the Random Number Generator (RNG) of the environment which is an np.random.RandomState object. This RNG should be used to perform calls to the desired random function to be used as noise to ensure reproducibility.
        reward_noise : float or Python function(rng)
            If it's a float, it's used as the standard deviation of an i.i.d. normal distribution of noise.
            If it's a Python function with one argument, it is added to the reward given at every time step. The argument is the Random Number Generator (RNG) of the environment which is an np.random.RandomState object. This RNG should be used to perform calls to the desired random function to be used as noise to ensure reproducibility.
        reward_density : float in range [0, 1]
            The fraction of possible sequences of a given length that will be selected to be rewardable at initialisation time.
        reward_scale : float
            Multiplies the rewards by this value at every time step.
        reward_shift : float
            This value is added to the reward at every time step.
        diameter : int > 0
            For discrete environments, if diameter = d, the set of states is set to be a d-partite graph (and NOT a complete d-partite graph), where, if we order the d sets as 1, 2, .., d, states from set 1 will have actions leading to states in set 2 and so on, with the final set d having actions leading to states in set 1. Number of actions for each state will, thus, be = (number of states) / (d).
        terminal_state_density : float in range [0, 1]
            For discrete environments, the fraction of states that are terminal; the terminal states are fixed to the "last" states when we consider them to be ordered by their numerical value. This is w.l.o.g. because discrete states are categorical. For continuous environments, please see terminal_states and term_state_edge for how to control terminal states.
        term_state_reward : float
            Adds this to the reward if a terminal state was reached at the current time step.
        image_representations : boolean
            Boolean to associate an image as the external observation with every discrete categorical state.
            For discrete envs, this is handled by an mdp_playground.spaces.ImageMultiDiscrete object. It associates the image of an n + 3 sided polygon for a categorical state n. More details can be found in the documentation for the ImageMultiDiscrete class.
            For continuous and grid envs, this is handled by an mdp_playground.spaces.ImageContinuous object. More details can be found in the documentation for the ImageContinuous class.
        irrelevant_features : boolean
            If True, an additional irrelevant sub-space (irrelevant to achieving rewards) is present as part of the observation space. This sub-space has its own transition dynamics independent of the dynamics of the relevant sub-space.
            For discrete environments, additionally, state_space_size must be specified as a list.
            For continuous environments, the option relevant_indices must be specified. This option specifies the dimensions relevant to achieving rewards.
            For grid environments, nothing additional needs to be done as relevant grid shape is also used as the irrelevant grid shape.
        use_custom_mdp : boolean
            If true, users specify their own transition and reward functions using the config options transition_function and reward_function (see below). Optionally, they can also use init_state_dist and terminal_states for discrete spaces (see below).
        transition_function : Python function(state, action) or a 2-D numpy.ndarray
            A Python function emulating P(s, a). For discrete envs it's also possible to specify an |S|x|A| transition matrix.
        reward_function : Python function(state_sequence, action_sequence) or a 2-D numpy.ndarray
            A Python function emulating R(state_sequence, action_sequence). The state_sequence is recorded by the environment and transition_function is called before reward_function, so the "current" state (when step() was called) and next state are the last 2 states in the sequence.
            For discrete environments, it's also possible to specify an |S|x|A| transition matrix where reward is assumed to be a function over the "current" state and action.
            If use_custom_mdp = false and the environment is continuous, this is a string that chooses one of the following predefined reward functions: move_along_a_line or move_to_a_point.
            If use_custom_mdp = false and the environment is grid, this is a string that chooses one of the following predefined reward functions: move_to_a_point. Support for sequences is planned.

            Also see make_denser documentation.


        Specific to discrete environments:
            state_space_size : int > 0 or list of length 2
                A number specifying size of the state space for normal discrete environments and a list of len = 2 when irrelevant_features is True (The list contains sizes of relevant and irrelevant sub-spaces where the 1st sub-space is assumed relevant and the 2nd sub-space is assumed irrelevant).
                NOTE: When automatically generating MDPs, do not specify this value as its value depends on the action_space_size and the diameter as state_space_size = action_space_size * diameter.
            action_space_size : int > 0
                Similar description as state_space_size. When automatically generating MDPs, however, its value determines the state_space_size.
            reward_dist : list with 2 floats or a Python function(env_rng, reward_sequence_dict)
                If it's a list with 2 floats, then these 2 values are interpreted as a closed interval and taken as the end points of a categorical distribution which points equally spaced along the interval.
                If it's a Python function, it samples rewards for the rewardable_sequences dict of the environment. The rewardable_sequences dict of the environment holds the rewardable_sequences with the key as a tuple holding the sequence and value as the reward handed out. The 1st argument for the reward_dist function is the Random Number Generator (RNG) of the environment which is an np.random.RandomState object. This RNG should be used to perform calls to the desired random function to be used to sample rewards to ensure reproducibility. The 2nd argument is the rewardable_sequences dict of the environment. This is available because one may need access to the already created reward sequences in the reward_dist function.
            init_state_dist : 1-D numpy.ndarray
                Specifies an array of initialisation probabilities for the discrete state space.
            terminal_states : Python function(state) or 1-D numpy.ndarray
                A Python function with the state as argument that returns whether the state is terminal. If this is specified as an array, the array lists the discrete states that are terminal.

            Specific to image_representations for discrete envs:
                image_transforms : str
                    String containing the transforms that must be applied to the image representations. As long as one of the following words is present in the string - shift, scale, rotate, flip - the corresponding transform will be applied at random to the polygon in the image representation whenever an observation is generated. Care is either explicitly taken that the polygon remains inside the image region or a warning is generated.
                sh_quant : int
                    An int to quantise the shift transforms.
                scale_range : (float, float)
                    A tuple of real numbers to specify (min_scaling, max_scaling).
                ro_quant : int
                    An int to quantise the rotation transforms.

        Specific to continuous environments:
            state_space_dim : int
                A number specifying state space dimensionality. A Gym Box space of this dimensionality will be instantiated.
            action_space_dim : int
                Same description as state_space_dim. This is currently set equal to the state_space_dim and doesn't need to specified.
            relevant_indices : list
                A list that provides the dimensions relevant to achieving rewards for continuous environments. The dynamics for these dimensions are independent of the dynamics for the remaining (irrelevant) dimensions.
            state_space_max : float
                Max absolute value that a dimension of the space can take. A Gym Box will be instantiated with range [-state_space_max, state_space_max]. Sampling will be done as for Gym Box spaces.
            action_space_max : float
                Similar description as for state_space_max.
            terminal_states : numpy.ndarray
                The centres of hypercube sub-spaces which are terminal.
            term_state_edge : float
                The edge of the hypercube sub-spaces which are terminal.
            transition_dynamics_order : int
                An order of n implies that the n-th state derivative is set equal to the action/inertia.
            inertia : float or numpy.ndarray
                inertia of the rigid body or point object that is being simulated. If numpy.ndarray, it specifies independent inertiae for the dimensions and the shape should be (state_space_dim,).
            time_unit : float
                time duration over which the action is applied to the system.
            target_point : numpy.ndarray
                The target point in case move_to_a_point is the reward_function. If make_denser is false, target_radius determines distance from the target point at which the sparse reward is handed out.
            action_loss_weight : float
                A coefficient to multiply the norm of the action and subtract it from the reward to penalise the action magnitude.

        Specific to grid environments:
            grid_shape : tuple
                Shape of the grid environment. If irrelevant_features is True, this is replicated to add a grid which is irrelevant to the reward.
            target_point : numpy.ndarray
                The target point in case move_to_a_point is the reward_function. If make_denser is false, reward is only handed out when the target point is reached.
            terminal_states : Python function(state) or 1-D numpy.ndarray
                Same description as for terminal_states under discrete envs

    Other important config:
        Specific to discrete environments:
            repeats_in_sequences : boolean
                If true, allows rewardable sequences to have repeating states in them.
            maximally_connected : boolean
                If true, sets the transition function such that every state in independent set i can transition to every state in independent set i + 1. If false, then sets the transition function such that a state in independent set i may have any state in independent set i + 1 as the next state for a transition.
            reward_every_n_steps : boolean
                Hand out rewards only at multiples of sequence_length steps. This makes the probability that an agent is executing overlapping rewarding sequences 0. This makes it simpler to evaluate HRL algorithms and whether they can "discretise" time correctly. Noise is added at every step, regardless of this setting. Currently, not implemented for either the make_denser = true case or for continuous and grid environments.
            generate_random_mdp : boolean
                If true, automatically generate MDPs when use_custom_mdp = false. Currently, this option doesn't need to be specified because random MDPs are always generated when use_custom_mdp = false.

        Specific to continuous environments:
            none as of now

        For all, continuous, discrete and grid environments:
        make_denser : boolean
            If true, makes the reward denser in environments.
            For discrete environments, hands out a partial reward for completing partial sequences.
            For continuous environments, for reward function move_to_a_point, the base reward handed out is equal to the distance moved towards the target point in the current timestep.
            For grid envs, the base reward handed out is equal to the Manhattan distance moved towards the target point in the current timestep.
        seed : int or dict
            Recommended to be passed as an int which generates seeds to be used for the various components of the environment. It is, however, possible to control individual seeds by passing it as a dict. Please see the default initialisation for seeds below to see how to do that.
        log_filename : str
            The name of the log file to which logs are written.
        log_level : logging.LOG_LEVEL option
            Python log level for logging

    Below, we list the important attributes and methods for this class.

    Attributes
    ----------
    config : dict
        the config contains all the details required to generate an environment
    seed : int or dict
        recommended to set to an int, which would set seeds for the env, relevant and irrelevant and externally visible observation and action spaces automatically. If fine-grained control over the seeds is necessary, a dict, with key values as in the source code further below, can be passed.
    observation_space : Gym.Space
        The externally visible observation space for the enviroment.
    action_space : Gym.Space
        The externally visible action space for the enviroment.
    rewardable_sequences : dict
        holds the rewardable sequences. The keys are tuples of rewardable sequences and values are the rewards handed out. When make_denser is True for discrete environments, this dict also holds the rewardable partial sequences.

    Methods
    -------
    init_terminal_states()
        Initialises terminal states, T
    init_init_state_dist()
        Initialises initial state distribution, rho_0
    init_transition_function()
        Initialises transition function, P
    init_reward_function()
        Initialises reward function, R
    transition_function(state, action)
        the transition function of the MDP, P
    P(state, action)
        defined as a lambda function in the call to init_transition_function() and is equivalent to calling transition_function()
    reward_function(state, action)
        the reward function of the MDP, R
    R(state, action)
        defined as a lambda function in the call to init_reward_function() and is equivalent to calling reward_function()
    get_augmented_state()
        gets underlying Markovian state of the MDP
    reset()
        Resets environment state
    seed()
        Sets the seed for the numpy RNG used by the environment (state and action spaces have their own seeds as well)
    step(action, imaginary_rollout=False)
        Performs 1 transition of the MDP
    """

    def __init__(self, **config):
        """Initialises the MDP to be emulated using the settings provided in config.

        Parameters
        ----------
        config : dict
            the member variable config is initialised to this value after inserting defaults
        """

        print("Passed config:", config, "\n")

        # Print initial "banner"
        screen_output_width = 132  # #hardcoded #TODO get from system
        repeat_equal_sign = (screen_output_width - 20) // 2
        set_ansi_escape = "\033[32;1m"
        reset_ansi_escape = "\033[0m"
        print(
            set_ansi_escape
            + "=" * repeat_equal_sign
            + "Initialising Toy MDP"
            + "=" * repeat_equal_sign
            + reset_ansi_escape
        )
        print("Current working directory:", os.getcwd())

        # Set other default settings for config to use if config is passed without any values for them
        if "log_level" not in config:
            self.log_level = logging.CRITICAL  # #logging.NOTSET
        else:
            self.log_level = config["log_level"]

        # print('self.log_level', self.log_level)
        logging.getLogger(__name__).setLevel(self.log_level)
        # fmtr = logging.Formatter(fmt='%(message)s - %(levelname)s - %(name)s - %(asctime)s', datefmt='%m.%d.%Y %I:%M:%S %p', style='%')
        # sh = logging.StreamHandler()
        # sh.setFormatter(fmt=fmtr)
        self.logger = logging.getLogger(__name__)
        # self.logger.addHandler(sh)

        if "log_filename" in config:
            #     self.log_filename = __name__ + '_' +\
            # datetime.today().strftime('%m.%d.%Y_%I:%M:%S_%f') + '.log' #
            # #TODO Make a directoy 'log/' and store there.
            # else:
            # checks that handlers is [], before adding a file logger, otherwise we
            # would have multiple loggers to file if multiple RLToyEnvs were
            # instantiated by the same process.
            if (not self.logger.handlers):
                self.log_filename = config["log_filename"]
                # logging.basicConfig(filename='/tmp/' + self.log_filename, filemode='a', format='%(message)s - %(levelname)s - %(name)s - %(asctime)s', datefmt='%m.%d.%Y %I:%M:%S %p', level=self.log_level)
                log_file_handler = logging.FileHandler(self.log_filename)
                self.logger.addHandler(log_file_handler)
        # log_filename = "logs/output.log"
        # os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        # #seed
        if (
            "seed" not in config
        ):  # #####IMP It's very important to not modify the config dict since it may be shared across multiple instances of the Env in the same process and could lead to very hard to catch bugs (I faced problems with Ray's A3C)
            self.seed_int = None
            need_to_gen_seeds = True
        elif isinstance(config["seed"], dict):
            self.seed_dict = config["seed"]
            need_to_gen_seeds = False
        elif (
            isinstance(config["seed"], int)
        ):  # should be an int then. Gym doesn't accept np.int64, etc..
            self.seed_int = config["seed"]
            need_to_gen_seeds = True
        else:
            raise TypeError("Unsupported data type for seed: ", type(config["seed"]))

        # #seed #TODO move to seed() so that obs., act. space, etc. have their
        # seeds reset too when env seed is reset?
        if need_to_gen_seeds:
            self.seed_dict = {}
            self.seed_dict["env"] = self.seed_int
            self.seed(self.seed_dict["env"])
            # ##IMP All these diff. seeds may not be needed (you could have one
            # seed for the joint relevant + irrelevant parts). But they allow for easy
            # separation of the relevant and irrelevant dimensions!! _And_ the seed
            # remaining the same for the underlying discrete environment makes it
            # easier to write tests!
            self.seed_dict["relevant_state_space"] = self.np_random.randint(
                sys.maxsize
            )  # #random
            self.seed_dict["relevant_action_space"] = self.np_random.randint(
                sys.maxsize
            )  # #random
            self.seed_dict["irrelevant_state_space"] = self.np_random.randint(
                sys.maxsize
            )  # #random
            self.seed_dict["irrelevant_action_space"] = self.np_random.randint(
                sys.maxsize
            )  # #random
            # #IMP This is currently used to sample only for continuous spaces and not used for discrete spaces by the Environment. User might want to sample from it for multi-discrete environments. #random
            self.seed_dict["state_space"] = self.np_random.randint(sys.maxsize)
            # #IMP This IS currently used to sample random actions by the RL agent for both discrete and continuous environments (but not used anywhere by the Environment). #random
            self.seed_dict["action_space"] = self.np_random.randint(sys.maxsize)
            self.seed_dict["image_representations"] = self.np_random.randint(
                sys.maxsize
            )  # #random
            # print("Mersenne0, dummy_eval:", self.np_random.get_state()[2], "dummy_eval" in config)
        else:  # if seed dict was passed
            self.seed(self.seed_dict["env"])
            # print("Mersenne0 (dict), dummy_eval:", self.np_random.get_state()[2], "dummy_eval" in config)

        self.logger.warning("Seeds set to:" + str(self.seed_dict))
        # print(f'Seeds set to {self.seed_dict=}') # Available from Python 3.8

        config["state_space_type"] = config["state_space_type"].lower()

        # #defaults ###TODO throw warning in case unknown config option is passed
        if "use_custom_mdp" not in config:
            self.use_custom_mdp = False
        else:
            self.use_custom_mdp = config["use_custom_mdp"]
            if self.use_custom_mdp:
                assert "transition_function" in config
                assert "reward_function" in config
                # if config["state_space_type"] == "discrete":
                #     assert "init_state_dist" in config

        if not self.use_custom_mdp:
            if "generate_random_mdp" not in config:
                self.generate_random_mdp = True
            else:
                self.generate_random_mdp = config["generate_random_mdp"]

        if "term_state_reward" not in config:
            self.term_state_reward = 0.0
        else:
            self.term_state_reward = config["term_state_reward"]

        if "delay" not in config:
            self.delay = 0
        else:
            self.delay = config["delay"]
        self.reward_buffer = [0.0] * (self.delay)

        if "sequence_length" not in config:
            self.sequence_length = 1
        else:
            self.sequence_length = config["sequence_length"]

        if "reward_density" not in config:
            self.reward_density = 0.25
        else:
            self.reward_density = config["reward_density"]

        if "make_denser" not in config:
            self.make_denser = False
        else:
            self.make_denser = config["make_denser"]

        if "maximally_connected" not in config:
            self.maximally_connected = True
        else:
            self.maximally_connected = config["maximally_connected"]

        if "reward_noise" in config:
            if callable(config["reward_noise"]):
                self.reward_noise = config["reward_noise"]
            else:
                reward_noise_std = config["reward_noise"]
                self.reward_noise = lambda a: a.normal(0, reward_noise_std)
        else:
            self.reward_noise = None

        if "transition_noise" in config:
            if config["state_space_type"] == "continuous":
                if callable(config["transition_noise"]):
                    self.transition_noise = config["transition_noise"]
                else:
                    p_noise_std = config["transition_noise"]
                    self.transition_noise = lambda a: a.normal(0, p_noise_std)
            else:  # discrete case
                self.transition_noise = config["transition_noise"]
        else:  # no transition noise
            self.transition_noise = None

        if "reward_scale" not in config:
            self.reward_scale = 1.0
        else:
            self.reward_scale = config["reward_scale"]

        if "reward_shift" not in config:
            self.reward_shift = 0.0
        else:
            self.reward_shift = config["reward_shift"]

        if "irrelevant_features" not in config:
            self.irrelevant_features = False
        else:
            self.irrelevant_features = config["irrelevant_features"]

        if "image_representations" not in config:
            self.image_representations = False
        else:
            self.image_representations = config["image_representations"]
            if "image_transforms" in config:
                assert config["state_space_type"] == "discrete", (
                    "Image " "transforms are only applicable to discrete envs."
                )
                self.image_transforms = config["image_transforms"]
            else:
                self.image_transforms = "none"

            if "image_width" in config:
                self.image_width = config["image_width"]
            else:
                self.image_width = 100

            if "image_height" in config:
                self.image_height = config["image_height"]
            else:
                self.image_height = 100

            # The following transforms are only applicable in discrete envs:
            if config["state_space_type"] == "discrete":
                if "image_sh_quant" not in config:
                    if "shift" in self.image_transforms:
                        warnings.warn(
                            "Setting image shift quantisation to the \
                        default of 1, since no config value was provided for it."
                        )
                        self.image_sh_quant = 1
                    else:
                        self.image_sh_quant = None
                else:
                    self.image_sh_quant = config["image_sh_quant"]

                if "image_ro_quant" not in config:
                    if "rotate" in self.image_transforms:
                        warnings.warn(
                            "Setting image rotate quantisation to the \
                        default of 1, since no config value was provided for it."
                        )
                        self.image_ro_quant = 1
                    else:
                        self.image_ro_quant = None
                else:
                    self.image_ro_quant = config["image_ro_quant"]

                if "image_scale_range" not in config:
                    if "scale" in self.image_transforms:
                        warnings.warn(
                            "Setting image scale range to the default \
                        of (0.5, 1.5), since no config value was provided for it."
                        )
                        self.image_scale_range = (0.5, 1.5)
                    else:
                        self.image_scale_range = None
                else:
                    self.image_scale_range = config["image_scale_range"]

        if config["state_space_type"] == "discrete":
            if "reward_dist" not in config:
                self.reward_dist = None
            else:
                self.reward_dist = config["reward_dist"]

            if "diameter" not in config:
                self.diameter = 1
            else:
                self.diameter = config["diameter"]

        elif config["state_space_type"] == "continuous":
            # if not self.use_custom_mdp:
            self.state_space_dim = config["state_space_dim"]

            if "transition_dynamics_order" not in config:
                self.dynamics_order = 1
            else:
                self.dynamics_order = config["transition_dynamics_order"]

            if "inertia" not in config:
                self.inertia = 1.0
            else:
                self.inertia = config["inertia"]

            if "time_unit" not in config:
                self.time_unit = 1.0
            else:
                self.time_unit = config["time_unit"]

            if "target_radius" not in config:
                self.target_radius = 0.05
            else:
                self.target_radius = config["target_radius"]

        elif config["state_space_type"] == "grid":
            assert "grid_shape" in config
            self.grid_shape = config["grid_shape"]
        else:
            raise ValueError("Unknown state_space_type")

        if "action_loss_weight" not in config:
            self.action_loss_weight = 0.0
        else:
            self.action_loss_weight = config["action_loss_weight"]

        if "reward_every_n_steps" not in config:
            self.reward_every_n_steps = False
        else:
            self.reward_every_n_steps = config["reward_every_n_steps"]

        if "repeats_in_sequences" not in config:
            self.repeats_in_sequences = False
        else:
            self.repeats_in_sequences = config["repeats_in_sequences"]

        self.dtype = np.float32 if "dtype" not in config else config["dtype"]

        if config["state_space_type"] == "discrete":
            if self.irrelevant_features:
                assert (
                    len(config["action_space_size"]) == 2
                ), "Currently, 1st sub-state (and action) space is assumed to be relevant to rewards and 2nd one is irrelevant. Please provide a list with sizes for the 2."
                self.action_space_size = config["action_space_size"]
            else:  # uni-discrete space
                assert isinstance(
                    config["action_space_size"], int
                ), "Did you mean to turn irrelevant_features? If so, please set irrelevant_features = True in config. If not, please provide an int for action_space_size."
                self.action_space_size = [
                    config["action_space_size"]
                ]  # Make a list to be able to iterate over observation spaces in for loops later
                # assert type(config["state_space_size"]) == int, 'config["state_space_size"] has to be provided as an int when we have a simple Discrete environment. Was:' + str(type(config["state_space_size"]))
            if self.use_custom_mdp:
                self.state_space_size = [config["state_space_size"]]
            else:
                self.state_space_size = np.array(self.action_space_size) * np.array(
                    self.diameter
                )
                # assert (np.array(self.state_space_size) % np.array(self.diameter) == 0).all(), "state_space_size should be a multiple of the diameter to allow for the generation of regularly connected MDPs."
        elif config["state_space_type"] == "continuous":
            self.action_space_dim = self.state_space_dim
            if self.irrelevant_features:
                assert (
                    "relevant_indices" in config
                ), "Please provide dimensions\
                 of state space relevant to rewards."
            if "relevant_indices" not in config:
                config["relevant_indices"] = range(self.state_space_dim)
            # config["irrelevant_indices"] = list(set(range(len(config["state_space_dim"]))) - set(config["relevant_indices"]))
        elif config["state_space_type"] == "grid":
            # Repeat the grid for the irrelevant part as well
            if self.irrelevant_features:
                self.grid_shape = self.grid_shape * 2

        if ("init_state_dist" in config) and ("relevant_init_state_dist" not in config):
            config["relevant_init_state_dist"] = config["init_state_dist"]

        assert (
            self.sequence_length > 0
        ), 'config["sequence_length"] <= 0. Set to: ' + str(
            self.sequence_length
        )  # also should be int
        if (
            "maximally_connected" in config and config["maximally_connected"]
        ):  # ###TODO remove
            pass
            # assert config["state_space_size"] == config["action_space_size"], "config[\"state_space_size\"] != config[\"action_space_size\"]. For maximally_connected transition graphs, they should be equal. Please provide valid values. Vals: " + str(config["state_space_size"]) + " " + str(config["action_space_size"]) + ". In future, \"maximally_connected\" graphs are planned to be supported!"
            # assert config["irrelevant_state_space_size"] ==
            # config["irrelevant_action_space_size"],
            # "config[\"irrelevant_state_space_size\"] !=
            # config[\"irrelevant_action_space_size\"]. For maximally_connected
            # transition graphs, they should be equal. Please provide valid values!
            # Vals: " + str(config["irrelevant_state_space_size"]) + " " +
            # str(config["irrelevant_action_space_size"]) + ". In future,
            # \"maximally_connected\" graphs are planned to be supported!" #TODO
            # Currently, irrelevant dimensions have a P similar to that of relevant
            # dimensions. Should this be decoupled?

        if config["state_space_type"] == "continuous":
            # assert config["state_space_dim"] == config["action_space_dim"], "For continuous spaces, state_space_dim has to be = action_space_dim. state_space_dim was: " + str(config["state_space_dim"]) + " action_space_dim was: " + str(config["action_space_dim"])
            if config["reward_function"] == "move_to_a_point":
                assert self.sequence_length == 1
                self.target_point = np.array(config["target_point"], dtype=self.dtype)
                assert self.target_point.shape == (
                    len(config["relevant_indices"]),
                ), "target_point should have dimensionality = relevant_state_space dimensionality"
        elif config["state_space_type"] == "grid":
            if config["reward_function"] == "move_to_a_point":
                self.target_point = config["target_point"]

        self.config = config
        self.augmented_state_length = self.sequence_length + self.delay + 1

        self.total_episodes = 0

        # This init_...() is done before the others below because it's needed
        # for image_representations for continuous
        self.init_terminal_states()

        if config["state_space_type"] == "discrete":
            self.observation_spaces = [
                DiscreteExtended(
                    self.state_space_size[0],
                    seed=self.seed_dict["relevant_state_space"],
                )
            ]  # #seed #hardcoded, many time below as well
            self.action_spaces = [
                DiscreteExtended(
                    self.action_space_size[0],
                    seed=self.seed_dict["relevant_action_space"],
                )
            ]  # #seed #hardcoded
            if self.irrelevant_features:
                self.observation_spaces.append(
                    DiscreteExtended(
                        self.state_space_size[1],
                        seed=self.seed_dict["irrelevant_state_space"],
                    )
                )  # #seed #hardcoded
                self.action_spaces.append(
                    DiscreteExtended(
                        self.action_space_size[1],
                        seed=self.seed_dict["irrelevant_action_space"],
                    )
                )  # #seed #hardcoded
            # Commented code below may used to generalise relevant sub-spaces to more than the current max of 2.
            # self.observation_spaces = [None] * len(config["all_indices"])
            # for i in config["relevant_indices"]:
            #     self.observation_spaces[i] =
            #     self.action_spaces[i] = DiscreteExtended(self.action_space_size[i], seed=self.seed_dict["relevant_action_space"]) #seed
            # for i in config["irrelevant_indices"]:
            #     self.observation_spaces[i] = DiscreteExtended(self.state_space_size[i], seed=self.seed_dict["irrelevant_state_space"])) #seed # hack
            # self.action_spaces[i] = DiscreteExtended(self.action_space_size[i],
            # seed=self.seed_dict["irrelevant_action_space"]) #seed

            if self.image_representations:
                # underlying_obs_space = MultiDiscreteExtended(self.state_space_size, seed=self.seed_dict["state_space"]) #seed
                self.observation_space = ImageMultiDiscrete(
                    self.state_space_size,
                    width=self.image_width,
                    height=self.image_height,
                    transforms=self.image_transforms,
                    sh_quant=self.image_sh_quant,
                    scale_range=self.image_scale_range,
                    ro_quant=self.image_ro_quant,
                    circle_radius=20,
                    seed=self.seed_dict["image_representations"],
                )  # #seed
                if self.irrelevant_features:
                    self.action_space = TupleExtended(
                        self.action_spaces, seed=self.seed_dict["action_space"]
                    )  # #seed
                else:
                    self.action_space = self.action_spaces[0]
            else:
                if self.irrelevant_features:
                    self.observation_space = TupleExtended(
                        self.observation_spaces, seed=self.seed_dict["state_space"]
                    )  # #seed # hack #TODO
                    # Gym (and so Ray) apparently needs observation_space as a
                    # member of an env.
                    self.action_space = TupleExtended(
                        self.action_spaces, seed=self.seed_dict["action_space"]
                    )  # #seed
                else:
                    self.observation_space = self.observation_spaces[0]
                    self.action_space = self.action_spaces[0]

        elif config["state_space_type"] == "continuous":
            self.state_space_max = (
                config["state_space_max"] if "state_space_max" in config else np.inf
            )  # should we
            # select a random max? #test?
            self.feature_space = BoxExtended(
                -self.state_space_max,
                self.state_space_max,
                shape=(self.state_space_dim,),
                seed=self.seed_dict["state_space"],
                dtype=self.dtype,
            )  # #seed
            # hack #TODO # low and high are 1st 2 and required arguments
            # for instantiating BoxExtended

            self.action_space_max = (
                config["action_space_max"] if "action_space_max" in config else np.inf
            )  # #test?
            # config["action_space_max"] = \
            # num_to_list(config["action_space_max"]) * config["action_space_dim"]
            self.action_space = BoxExtended(
                -self.action_space_max,
                self.action_space_max,
                shape=(self.action_space_dim,),
                seed=self.seed_dict["action_space"],
                dtype=self.dtype,
            )  # #seed
            # hack #TODO

            if self.image_representations:
                self.observation_space = ImageContinuous(
                    self.feature_space,
                    width=self.image_width,
                    height=self.image_height,
                    term_spaces=self.term_spaces,
                    target_point=self.target_point,
                    circle_radius=5,
                    seed=self.seed_dict["image_representations"],
                )  # #seed
            else:
                self.observation_space = self.feature_space

        elif config["state_space_type"] == "grid":
            underlying_space_maxes = list_to_float_np_array(self.grid_shape)

            # The min for grid envs is 0, 0, 0, ...
            self.feature_space = BoxExtended(
                0 * underlying_space_maxes,
                underlying_space_maxes,
                seed=self.seed_dict["state_space"],
                dtype=self.dtype,
            )  # #seed

            lows = np.array([-1] * len(self.grid_shape))
            highs = np.array([1] * len(self.grid_shape))
            self.action_space = GridActionSpace(
                lows,
                highs,
                seed=self.seed_dict["action_space"],
            )  # #seed

            if self.image_representations:
                target_pt = list_to_float_np_array(self.target_point)
                self.observation_space = ImageContinuous(
                    self.feature_space,
                    width=self.image_width,
                    height=self.image_height,
                    term_spaces=self.term_spaces,
                    target_point=target_pt,
                    circle_radius=5,
                    grid_shape=self.grid_shape,
                    seed=self.seed_dict["image_representations"],
                )  # #seed
            else:
                self.observation_space = self.feature_space

        # if config["action_space_type"] == "discrete":
        #     if not config["generate_random_mdp"]:
        #         # self.logger.error("User defined P and R are currently not supported.") ##TODO
        #         # sys.exit(1)
        #         self.P = config["transition_function"] if callable(config["transition_function"]) else lambda s, a: config["transition_function"][s, a] ##IMP callable may not be optimal always since it was deprecated in Python 3.0 and 3.1
        #         self.R = config["reward_function"] if callable(config["reward_function"]) else lambda s, a: config["reward_function"][s, a]
        # else:
        # ##TODO Support imaginary rollouts for continuous envs. and user-defined P and R? Will do it depending on demand for it. In fact, for imagined rollouts, let our code handle storing augmented_state, curr_state, etc. in separate variables, so that it's easy for user to perform imagined rollouts instead of having to maintain their own state and action sequences.
        # #TODO Generate state and action space sizes also randomly?

        # ###IMP The order in which the following inits are called is important, so don't change!!
        # #init_state_dist: Initialises uniform distribution over non-terminal states for discrete distribution; After looking into Gym code, I can say that for continuous, it's uniform over non-terminal if limits are [a, b], shifted exponential if exactly one of the limits is np.inf, normal if both limits are np.inf - this sampling is independent for each dimension (and is done for the defined limits for the respective dimension).
        self.init_init_state_dist()
        self.init_transition_function()
        # print("Mersenne1, dummy_eval:", self.np_random.get_state()[2], "dummy_eval" in self.config)
        self.init_reward_function()

        self.curr_obs = (
            self.reset()
        )  # #TODO Maybe not call it here, since Gym seems to expect to _always_ call this method when using an environment; make this seedable? DO NOT do seed dependent initialization in reset() otherwise the initial state distrbution will always be at the same state at every call to reset()!! (Gym env has its own seed? Yes, it does, as does also space);

        self.logger.info(
            "self.augmented_state, len: "
            + str(self.augmented_state)
            + ", "
            + str(len(self.augmented_state))
        )
        self.logger.info(
            "MDP Playground toy env instantiated with config: " + str(self.config)
        )
        print("MDP Playground toy env instantiated with config: " + str(self.config))

    def init_terminal_states(self):
        """Initialises terminal state set to be the 'last' states for discrete environments. For continuous environments, terminal states will be in a hypercube centred around config['terminal_states'] with the edge of the hypercube of length config['term_state_edge']."""
        if self.config["state_space_type"] == "discrete":
            if (
                self.use_custom_mdp and "terminal_state_density" not in self.config
            ):  # custom/user-defined terminal states
                self.is_terminal_state = (
                    self.config["terminal_states"]
                    if callable(self.config["terminal_states"])
                    else lambda s: s in self.config["terminal_states"]
                )
            else:
                # Define the no. of terminal states per independent set of the state space
                self.num_terminal_states = int(
                    self.config["terminal_state_density"] * self.action_space_size[0]
                )  # #hardcoded ####IMP Using action_space_size
                # since it contains state_space_size // diameter
                # if self.num_terminal_states == 0: # Have at least 1 terminal state?
                #     warnings.warn("WARNING: int(terminal_state_density * relevant_state_space_size) was 0. Setting num_terminal_states to be 1!")
                #     self.num_terminal_states = 1
                self.config["terminal_states"] = np.array(
                    [
                        j * self.action_space_size[0] - 1 - i
                        for j in range(1, self.diameter + 1)
                        for i in range(self.num_terminal_states)
                    ]
                )  # terminal states
                # inited to be at the "end" of the sorted states
                self.logger.warning(
                    "Inited terminal states to self.config['terminal_states']: "
                    + str(self.config["terminal_states"])
                    + ". Total "
                    + str(self.num_terminal_states)
                )
                self.is_terminal_state = lambda s: s in self.config["terminal_states"]

        elif self.config["state_space_type"] == "continuous":
            # print("# TODO for cont. spaces: term states")
            self.term_spaces = []

            if "terminal_states" in self.config:  # ##TODO For continuous spaces,
                # could also generate terminal spaces based on a terminal_state_density
                # given by user (Currently, user specifies terminal state points
                # around which hypercubes in state space are terminal. If the user
                # want a specific density and not hypercubes, the user has to design
                # the terminal states they specify such that they would have a given
                # density in space.). But only for state spaces with limits? For state
                # spaces without limits, could do it for a limited subspace of the
                # infinite state space 1st and then repeat that pattern indefinitely
                # along each dimension's axis. #test?
                if callable(self.config["terminal_states"]):
                    self.is_terminal_state = self.config["terminal_states"]
                else:
                    for i in range(
                        len(self.config["terminal_states"])
                    ):  # List of centres
                        # of terminal state regions.
                        assert len(self.config["terminal_states"][i]) == len(
                            self.config["relevant_indices"]
                        ), (
                            "Specified terminal state centres should have"
                            " dimensionality = number of relevant_indices. That"
                            " was not the case for centre no.: " + str(i) + ""
                        )
                        lows = np.array(
                            [
                                self.config["terminal_states"][i][j]
                                - self.config["term_state_edge"] / 2
                                for j in range(len(self.config["relevant_indices"]))
                            ]
                        )
                        highs = np.array(
                            [
                                self.config["terminal_states"][i][j]
                                + self.config["term_state_edge"] / 2
                                for j in range(len(self.config["relevant_indices"]))
                            ]
                        )
                        # print("Term state lows, highs:", lows, highs)
                        self.term_spaces.append(
                            BoxExtended(
                                low=lows, high=highs, seed=self.seed_, dtype=self.dtype
                            )
                        )  # #seed #hack #TODO
                    self.logger.debug(
                        "self.term_spaces samples:"
                        + str(self.term_spaces[0].sample())
                        + str(self.term_spaces[-1].sample())
                    )

                    self.is_terminal_state = lambda s: np.any(
                        [
                            self.term_spaces[i].contains(
                                s[self.config["relevant_indices"]]
                            )
                            for i in range(len(self.term_spaces))
                        ]
                    )
                    # ### TODO for cont. #test?

            else:  # no custom/user-defined terminal states
                self.is_terminal_state = lambda s: False

        elif self.config["state_space_type"] == "grid":
            self.term_spaces = []

            if "terminal_states" in self.config:
                if callable(self.config["terminal_states"]):
                    self.is_terminal_state = self.config["terminal_states"]
                else:
                    for i in range(len(self.config["terminal_states"])):  # List of
                        # terminal states on the grid
                        term_state = list_to_float_np_array(
                            self.config["terminal_states"][i]
                        )
                        lows = term_state
                        highs = term_state  # #hardcoded
                        self.term_spaces.append(
                            BoxExtended(
                                low=lows, high=highs, seed=self.seed_, dtype=np.int64
                            )
                        )  # #seed #hack #TODO

                    def is_term(s):
                        cont_state = list_to_float_np_array(s)
                        return np.any(
                            [
                                self.term_spaces[i].contains(cont_state)
                                for i in range(len(self.term_spaces))
                            ]
                        )

                    self.is_terminal_state = is_term

            else:  # no custom/user-defined terminal states
                self.is_terminal_state = lambda s: False

    def init_init_state_dist(self):
        """Initialises initial state distrbution, rho_0, to be uniform over the non-terminal states for discrete environments. For both discrete and continuous environments, the uniform sampling over non-terminal states is taken care of in reset() when setting the initial state for an episode."""
        # relevant dimensions part
        if self.config["state_space_type"] == "discrete":
            if (
                self.use_custom_mdp and "init_state_dist" in self.config
            ):  # custom/user-defined phi_0
                # self.config["relevant_init_state_dist"] = #TODO make this also a lambda function?
                pass
            else:
                # For relevant sub-space
                non_term_state_space_size = (
                    self.action_space_size[0] - self.num_terminal_states
                )  # #hardcoded
                self.config["relevant_init_state_dist"] = (
                    [
                        1 / (non_term_state_space_size * self.diameter)
                        for i in range(non_term_state_space_size)
                    ]
                    + [0 for i in range(self.num_terminal_states)]
                ) * self.diameter  # #TODO
                # Currently only uniform distribution over non-terminal
                # states; Use Dirichlet distribution to select prob. distribution to use?
                # #TODO make init_state_dist the default sample() for state space?
                self.config["relevant_init_state_dist"] = np.array(
                    self.config["relevant_init_state_dist"]
                )
                self.logger.warning(
                    "self.relevant_init_state_dist:"
                    + str(self.config["relevant_init_state_dist"])
                )

                # #irrelevant sub-space
                if self.irrelevant_features:
                    non_term_state_space_size = self.state_space_size[1]  # #hardcoded
                    self.config["irrelevant_init_state_dist"] = [
                        1 / (non_term_state_space_size)
                        for i in range(non_term_state_space_size)
                    ]  # diameter not needed here as we directly take the state_space_size in the prev. line
                    self.config["irrelevant_init_state_dist"] = np.array(
                        self.config["irrelevant_init_state_dist"]
                    )
                    self.logger.warning(
                        "self.irrelevant_init_state_dist:"
                        + str(self.config["irrelevant_init_state_dist"])
                    )

        else:  # if continuous or grid space
            pass  # this is handled in reset where we resample if we sample a term. state

    def init_transition_function(self):
        """Initialises transition function, P by selecting random next states for every (state, action) tuple for discrete environments. For continuous environments, we have 1 option for the transition function which varies depending on dynamics order and inertia and time_unit for a point object."""

        if self.config["state_space_type"] == "discrete":
            if self.use_custom_mdp:  # custom/user-defined P
                pass
            else:
                # relevant dimensions part
                self.config["transition_function"] = np.zeros(
                    shape=(self.state_space_size[0], self.action_space_size[0]),
                    dtype=object,
                )  # #hardcoded
                self.config["transition_function"][:] = -1  # #IMP # To avoid
                # having a valid value from the state space before we actually
                # assign a usable value below!
                if self.maximally_connected:
                    if self.diameter == 1:  # #hack # TODO Remove this if block;
                        # this case is currently separately handled just so that tests
                        # do not fail. Using prob=prob in the sample call causes the
                        # sampling to change even if the probabilities remain the
                        # same. All solutions I can think of are hacky except changing
                        # the expected values in all the test cases which would take
                        # quite some time.
                        for s in range(self.state_space_size[0]):
                            self.config["transition_function"][
                                s
                            ] = self.observation_spaces[0].sample(
                                size=self.action_space_size[0], replace=False
                            )  # #random #TODO Preferably use the seed of the
                            # Env for this? #hardcoded
                    else:  # if diam > 1
                        for s in range(self.state_space_size[0]):
                            i_s = (
                                s // self.action_space_size[0]
                            )  # select the current independent set number

                            prob = np.zeros(shape=(self.state_space_size[0],))
                            prob_next_states = (
                                np.ones(shape=(self.action_space_size[0],))
                                / self.action_space_size[0]
                            )
                            ind_1 = (
                                (i_s + 1) * self.action_space_size[0]
                            ) % self.state_space_size[0]
                            ind_2 = (
                                (i_s + 2) * self.action_space_size[0]
                            ) % self.state_space_size[0]
                            # print(ind_1, ind_2)
                            if ind_2 <= ind_1:  # edge case
                                ind_2 += self.state_space_size[0]
                            prob[ind_1:ind_2] = prob_next_states
                            self.config["transition_function"][
                                s
                            ] = self.observation_spaces[0].sample(
                                prob=prob, size=self.action_space_size[0], replace=False
                            )  # #random #TODO
                            # Preferably use the seed of the Env for this? #hardcoded

                            # hacky way to do the above
                            # self.config["transition_function"][s] = self.observation_spaces[0].sample(max=self.action_space_size[0], size=self.action_space_size[0], replace=False) #random #TODO Preferably use the seed of the Env for this? #hardcoded
                            # Set the transitions from current state to be to the next independent set's states
                            # self.config["transition_function"][s] += ((i_s + 1) * self.action_space_size[0]) % self.state_space_size[0]
                else:  # if not maximally_connected
                    for s in range(self.state_space_size[0]):
                        i_s = (
                            s // self.action_space_size[0]
                        )  # select the current independent
                        # set number

                        # Set the probabilities of the next state for the current independent set
                        prob = np.zeros(shape=(self.state_space_size[0],))
                        prob_next_states = (
                            np.ones(shape=(self.action_space_size[0],))
                            / self.action_space_size[0]
                        )
                        ind_1 = (
                            (i_s + 1) * self.action_space_size[0]
                        ) % self.state_space_size[0]
                        ind_2 = (
                            (i_s + 2) * self.action_space_size[0]
                        ) % self.state_space_size[0]
                        # print(ind_1, ind_2)
                        if ind_2 <= ind_1:  # edge case
                            ind_2 += self.state_space_size[0]
                        prob[ind_1:ind_2] = prob_next_states

                        for a in range(self.action_space_size[0]):
                            # prob[i_s * self.action_space_size[0] : (i_s + 1) * self.action_space_size[0]] = prob_next_states
                            self.config["transition_function"][
                                s, a
                            ] = self.observation_spaces[0].sample(prob=prob)
                            # #random #TODO Preferably use the seed of the Env for this?
                # Set the next state for terminal states to be themselves, for any action taken.
                for i_s in range(self.diameter):
                    for s in range(
                        self.action_space_size[0] - self.num_terminal_states,
                        self.action_space_size[0],
                    ):
                        for a in range(self.action_space_size[0]):
                            assert (
                                self.is_terminal_state(
                                    i_s * self.action_space_size[0] + s
                                )

                            )
                            self.config["transition_function"][
                                i_s * self.action_space_size[0] + s, a
                            ] = (
                                i_s * self.action_space_size[0] + s
                            )  # Setting
                            # P(s, a) = s for terminal states, for P() to be
                            # meaningful even if someone doesn't check for
                            # 'done' being = True

                # #irrelevant dimensions part
                if self.irrelevant_features:  # #test
                    self.config["transition_function_irrelevant"] = np.zeros(
                        shape=(self.state_space_size[1], self.action_space_size[1]),
                        dtype=object,
                    )
                    self.config["transition_function_irrelevant"][:] = -1  # #IMP
                    # To avoid having a valid value from the state space before we
                    # actually assign a usable value below!
                    if self.maximally_connected:
                        for s in range(self.state_space_size[1]):
                            i_s = s // self.action_space_size[1]  # select the
                            # current independent set number

                            # Set the probabilities of the next state for the
                            # current independent set
                            prob = np.zeros(shape=(self.state_space_size[1],))
                            prob_next_states = (
                                np.ones(shape=(self.action_space_size[1],))
                                / self.action_space_size[1]
                            )
                            ind_1 = (
                                (i_s + 1) * self.action_space_size[1]
                            ) % self.state_space_size[1]
                            ind_2 = (
                                (i_s + 2) * self.action_space_size[1]
                            ) % self.state_space_size[1]
                            print(ind_1, ind_2)
                            if ind_2 <= ind_1:  # edge case
                                ind_2 += self.state_space_size[1]
                            prob[ind_1:ind_2] = prob_next_states

                            self.config["transition_function_irrelevant"][
                                s
                            ] = self.observation_spaces[1].sample(
                                prob=prob, size=self.action_space_size[1], replace=False
                            )
                            # #random #TODO Preferably use the seed of the
                            # Env for this? #hardcoded

                            # self.config["transition_function_irrelevant"][s] = self.observation_spaces[1].sample(max=self.action_space_size[1], size=self.action_space_size[1], replace=False) #random #TODO Preferably use the seed of the Env for this?
                            # self.config["transition_function_irrelevant"][s] += ((i_s + 1) * self.action_space_size[1]) % self.state_space_size[1]
                    else:
                        for s in range(self.state_space_size[1]):
                            i_s = s // self.action_space_size[1]  # select the
                            # current independent set number

                            # Set the probabilities of the next state for the
                            # current independent set
                            prob = np.zeros(shape=(self.state_space_size[1],))
                            prob_next_states = (
                                np.ones(shape=(self.action_space_size[1],))
                                / self.action_space_size[1]
                            )
                            ind_1 = (
                                (i_s + 1) * self.action_space_size[1]
                            ) % self.state_space_size[1]
                            ind_2 = (
                                (i_s + 2) * self.action_space_size[1]
                            ) % self.state_space_size[1]
                            # print(ind_1, ind_2)
                            if ind_2 <= ind_1:  # edge case
                                ind_2 += self.state_space_size[1]
                            prob[ind_1:ind_2] = prob_next_states

                            for a in range(self.action_space_size[1]):
                                # prob[i_s * self.action_space_size[1] : (i_s + 1)
                                # * self.action_space_size[1]] = prob_next_states
                                self.config["transition_function_irrelevant"][
                                    s, a
                                ] = self.observation_spaces[1].sample(prob=prob)
                                # #random #TODO Preferably use the seed of the Env for this?

                    self.logger.warning(
                        str(self.config["transition_function_irrelevant"])
                        + "init_transition_function _irrelevant"
                        + str(type(self.config["transition_function_irrelevant"][0, 0]))
                    )

            if not callable(self.config["transition_function"]):
                self.transition_matrix = self.config["transition_function"]
                self.config[
                    "transition_function"
                ] = lambda s, a: self.transition_matrix[s, a]
                print(
                    "transition_matrix inited to:\n"
                    + str(self.transition_matrix)
                    + "\nPython type of state: "
                    + str(type(self.config["transition_function"](0, 0)))
                )  # The
                # Python type of the state can lead to hard to catch bugs

        else:  # if continuous or grid space
            # self.logger.debug("# TODO for cont. spaces") # transition function is a
            # fixed parameterisation for cont. envs. right now.
            pass

        self.P = lambda s, a: self.transition_function(s, a)

    def init_reward_function(self):
        """Initialises reward function, R by selecting random sequences to be rewardable for discrete environments. For continuous environments, we have fixed available options for the reward function."""
        # print("Mersenne2, dummy_eval:", self.np_random.get_state()[2], "dummy_eval" in self.config)

        # #TODO Maybe refactor this code and put useful reusable permutation generators, etc. in one library
        if self.config["state_space_type"] == "discrete":
            if self.use_custom_mdp:  # custom/user-defined R
                if not callable(self.config["reward_function"]):
                    self.reward_matrix = self.config["reward_function"]
                    self.config["reward_function"] = lambda s, a: self.reward_matrix[
                        s[-2], a
                    ]  # #hardcoded
                    # to be 2nd last state in state sequence passed to reward
                    # function, so that reward is R(s, a) when transition is s, a, r, s'
                    print("reward_matrix inited to:" + str(self.reward_matrix))
            else:
                non_term_state_space_size = (
                    self.action_space_size[0] - self.num_terminal_states
                )

                def get_sequences(maximum, length, fraction, repeats=False, diameter=1):
                    """
                    Returns random sequences of integers

                    maximum: int
                        Max value of the integers in the sequence
                    length: int
                        Length of sequence
                    fraction: float
                        Fraction of total possible sequences to be returned
                    repeats: boolean
                        Allows repeats in returned sequences
                    diameter: int
                        Relates to the diameter of the MDP
                    """

                    sequences = []

                    if repeats:
                        num_possible_sequences = (maximum) ** length
                        num_sel_sequences = int(fraction * num_possible_sequences)
                        if num_sel_sequences == 0:
                            num_sel_sequences = 1
                            warnings.warn(
                                "0 rewardable sequences per independent"
                                " set for given reward_density, sequence_length,"
                                " diameter and terminal_state_density. Setting it to 1."
                            )
                        sel_sequence_nums = self.np_random.choice(
                            num_possible_sequences,
                            size=num_sel_sequences,
                            replace=False,
                        )  # #random # This assumes that all
                        # sequences have an equal likelihood of being selected
                        # for being a reward sequence; This line also makes it
                        # not possible to have this function be portable as
                        # part of a library because it use the np_random
                        # member variable of this class
                        for i_s in range(diameter):  # Allow sequences to begin in
                            # any of the independent sets and therefore this loop is
                            # over the no. of independent sets(= diameter)
                            for i in range(num_sel_sequences):
                                curr_sequence_num = sel_sequence_nums[i]
                                specific_sequence = []
                                while len(specific_sequence) != length:
                                    specific_sequence.append(
                                        curr_sequence_num % (non_term_state_space_size)
                                        + ((len(specific_sequence) + i_s) % diameter)
                                        * self.action_space_size[0]
                                    )
                                    # #TODO this uses a member variable of the
                                    # class. Add another function param to
                                    # receive this value? Name it independent set size?
                                    curr_sequence_num = curr_sequence_num // (
                                        non_term_state_space_size
                                    )
                                    # #bottleneck When we sample sequences here,
                                    # it could get very slow if reward_density is
                                    # high; alternative would be to assign numbers
                                    # to sequences and then sample these numbers
                                    # without replacement and take those sequences
                                    # specific_sequence =
                                    # self.relevant_observation_space.sample(size=self.sequence_length,
                                    # replace=True) # Be careful that sequence_length is less than state space
                                    # size
                                sequences.append(specific_sequence)
                            self.logger.info(
                                "Total no. of rewarded sequences:"
                                + str(len(sequences))
                                + "Out of"
                                + str(num_possible_sequences)
                                + "per independent set"
                            )
                    else:  # if no repeats
                        assert length <= diameter * maximum, (
                            "When there are no"
                            " repeats in sequences, the sequence length should be"
                            " <= diameter * maximum."
                        )
                        permutations = []
                        for i in range(length):
                            permutations.append(maximum - i // diameter)
                        # permutations = list(range(maximum + 1 - length, maximum + 1))
                        self.logger.info(
                            "No. of choices for each element in a"
                            " possible sequence (Total no. of permutations will be a"
                            " product of this), no. of possible perms per independent"
                            " set: "
                            + str(permutations)
                            + ", "
                            + str(np.prod(permutations))
                        )

                        for i_s in range(diameter):  # Allow sequences to begin in
                            # any of the independent sets and therefore this loop is
                            # over the no. of independent sets(= diameter). Could
                            # maybe sample independent set no. as "part" of
                            # sel_sequence_nums below and avoid this loop?
                            num_possible_permutations = np.prod(permutations)  # Number
                            # of possible permutations/sequences for, say, a
                            # diameter of 3 and 24 total states and
                            # terminal_state_density = 0.25, i.e., 6 non-terminal
                            # states (out of 8 states) per independent set, for
                            # sequence length of 5 is np.prod([6, 6, 6, 5, 5]) * 3;
                            # the * diameter at the end is needed because the
                            # sequence can begin in any of the independent sets;
                            # However, for simplicity, we omit * diameter here and
                            # just perform the same procedure per independent set.
                            # This can lead to slightly fewer rewardable sequences
                            # than should be the case for a given reward_density -
                            # this is due int() in the next step
                            num_sel_sequences = int(
                                fraction * num_possible_permutations
                            )
                            if (
                                num_sel_sequences == 0
                            ):  # ##TODO Remove this test here and above?
                                num_sel_sequences = 1
                                warnings.warn(
                                    "0 rewardable sequences per"
                                    " independent set for given reward_density,"
                                    " sequence_length, diameter and"
                                    " terminal_state_density. Setting it to 1."
                                )
                            # print("Mersenne3:", self.np_random.get_state()[2])
                            sel_sequence_nums = self.np_random.choice(
                                num_possible_permutations,
                                size=num_sel_sequences,
                                replace=False,
                            )  # #random # This assumes that all
                            # sequences have an equal likelihood of being
                            # selected for being a reward sequence; # TODO
                            # this code could be replaced with self.np_random.permutation(
                            # non_term_state_space_size)[self.sequence_length]?
                            # Replacement becomes a problem then! We have to
                            # keep sampling until we have all unique rewardable sequences.
                            # print("Mersenne4:", self.np_random.get_state()[2])

                            total_clashes = 0
                            for i in range(num_sel_sequences):
                                curr_permutation = sel_sequence_nums[i]
                                seq_ = []
                                curr_rem_digits = []
                                for j in range(diameter):
                                    curr_rem_digits.append(
                                        list(range(maximum))
                                    )  # # has to contain every number up to n so
                                    # that any one of them can be picked as part
                                    # of the sequence below
                                for enum, j in enumerate(permutations):  # Goes
                                    # from largest to smallest number among the factors of nPk
                                    rem_ = curr_permutation % j
                                    # rem_ = (enum // maximum) * maximum + rem_
                                    seq_.append(
                                        curr_rem_digits[(enum + i_s) % diameter][rem_]
                                        + ((enum + i_s) % diameter)
                                        * self.action_space_size[0]
                                    )  # Use (enum + i_s)
                                    # to allow other independent sets to have
                                    # states beginning a rewardable sequence
                                    del curr_rem_digits[(enum + i_s) % diameter][rem_]
                                    #         print("curr_rem_digits", curr_rem_digits)
                                    curr_permutation = curr_permutation // j

                                if seq_ in sequences:  # #hack
                                    total_clashes += (
                                        1  # #TODO remove these extra checks and
                                    )
                                    # assert below
                                sequences.append(seq_)

                            self.logger.debug(
                                "Number of generated sequences that"
                                " did not clash with an existing one when it was"
                                " generated:" + str(total_clashes)
                            )
                            assert total_clashes == 0, (
                                "None of the generated"
                                " sequences should have clashed with an existing"
                                " rewardable sequence when it was generated. No. of"
                                " times a clash was detected:" + str(total_clashes)
                            )
                            self.logger.info(
                                "Total no. of rewarded sequences:"
                                + str(len(sequences))
                                + "Out of"
                                + str(num_possible_permutations)
                                + "per independent set"
                            )

                    return sequences

                def insert_sequence(sequence):
                    """
                    Inserts rewardable sequences into the rewardable_sequences dict member variable
                    """
                    sequence = tuple(sequence)  # tuples are immutable and can be
                    # used as keys for a dict
                    if callable(self.reward_dist):
                        self.rewardable_sequences[sequence] = self.reward_dist(
                            self.np_random, self.rewardable_sequences
                        )
                    else:
                        self.rewardable_sequences[sequence] = 1.0  # this is the
                        # default reward value, reward scaling will be handled later
                    self.logger.warning(
                        "specific_sequence that will be rewarded" + str(sequence)
                    )
                    # #TODO impose a different distribution for these:
                    # independently sample state for each step of specific
                    # sequence; or conditionally dependent samples if we want
                    # something like DMPs/manifolds
                    if self.make_denser:
                        for ss_len in range(1, len(sequence)):
                            sub_sequence = tuple(sequence[:ss_len])
                            if sub_sequence not in self.rewardable_sequences:
                                self.rewardable_sequences[sub_sequence] = 0.0
                            self.rewardable_sequences[sub_sequence] += (
                                self.rewardable_sequences[sequence]
                                * ss_len
                                / len(sequence)
                            )
                            # this could cause problems if we support variable sequence lengths and
                            # there are clashes in selected rewardable sequences

                self.rewardable_sequences = {}
                if self.repeats_in_sequences:
                    rewardable_sequences = get_sequences(
                        maximum=non_term_state_space_size,
                        length=self.sequence_length,
                        fraction=self.reward_density,
                        repeats=True,
                        diameter=self.diameter,
                    )

                else:  # if no repeats_in_sequences
                    rewardable_sequences = get_sequences(
                        maximum=non_term_state_space_size,
                        length=self.sequence_length,
                        fraction=self.reward_density,
                        repeats=False,
                        diameter=self.diameter,
                    )

                # Common to both cases: repeats_in_sequences or not
                if isinstance(self.reward_dist, list):  # Specified as interval
                    reward_dist_ = self.reward_dist
                    num_rews = self.diameter * len(rewardable_sequences)
                    print("num_rewardable_sequences set to:", num_rews)
                    if num_rews == 1:
                        rews = [1.0]
                    else:
                        rews = np.linspace(
                            reward_dist_[0], reward_dist_[1], num=num_rews
                        )
                    assert rews[-1] == 1.0
                    self.np_random.shuffle(rews)

                    def get_rews(rng, r_dict):
                        return rews[len(r_dict)]

                    self.reward_dist = get_rews

                if len(rewardable_sequences) > 1000:
                    warnings.warn(
                        "Too many rewardable sequences and/or too long"
                        " rewardable sequence length. Environment might be too slow."
                        " Please consider setting the reward_density to be lower or"
                        " reducing the sequence length. No. of rewardable sequences:"
                        + str(len(rewardable_sequences))
                    )  # #TODO Maybe even exit the
                    # program if too much memory is (expected to be) taken.; Took
                    # about 80s for 40k iterations of the for loop below on my laptop

                for specific_sequence in rewardable_sequences:
                    insert_sequence(specific_sequence)
                # else: # "repeats" in sequences are allowed until diameter - 1
                # steps have been taken: We sample the sequences as the state
                # number inside each independent set, which are numbered from 0 to
                # action_space_size - 1
                #     pass

                print(
                    "rewardable_sequences: " + str(self.rewardable_sequences)
                )  # #debug print
        elif self.config["state_space_type"] == "continuous":
            # self.logger.debug("# TODO for cont. spaces?: init_reward_function")
            # reward functions are fixed for cont. right now with a few available choices.
            pass
        elif self.config["state_space_type"] == "grid":
            ...  # ###TODO Make sequences compatible with grid

        self.R = lambda s, a: self.reward_function(s, a)

    def transition_function(self, state, action):
        """The transition function, P.

        Performs a transition according to the initialised P for discrete environments (with dynamics independent for relevant vs irrelevant dimension sub-spaces). For continuous environments, we have a fixed available option for the dynamics (which is the same for relevant or irrelevant dimensions):
        The order of the system decides the dynamics. For an nth order system, the nth order derivative of the state is set to the action value / inertia for time_unit seconds. And then the dynamics are integrated over the time_unit to obtain the next state.

        Parameters
        ----------
        state : list
            The state that the environment will use to perform a transition.
        action : list
            The action that the environment will use to perform a transition.

        Returns
        -------
        int or np.array
            The state at the end of the current transition
        """

        if self.config["state_space_type"] == "discrete":
            next_state = self.config["transition_function"](state, action)
            if self.transition_noise:
                probs = (
                    np.ones(shape=(self.state_space_size[0],))
                    * self.transition_noise
                    / (self.state_space_size[0] - 1)
                )
                probs[next_state] = 1 - self.transition_noise
                # TODO Samples according to new probs to get noisy discrete transition
                new_next_state = self.observation_spaces[0].sample(prob=probs)  # random
                # print("noisy old next_state, new_next_state", next_state, new_next_state)
                if next_state != new_next_state:
                    self.logger.info(
                        "NOISE inserted! old next_state, new_next_state"
                        + str(next_state)
                        + str(new_next_state)
                    )
                    self.total_noisy_transitions_episode += 1
                # print("new probs:", probs, self.relevant_observation_space.sample(prob=probs))
                next_state = new_next_state
                # assert np.sum(probs) == 1, str(np.sum(probs)) + " is not equal to " + str(1)

        elif self.config["state_space_type"] == "continuous":
            # ##TODO implement imagined transitions also for cont. spaces
            if self.use_custom_mdp:
                next_state = self.config["transition_function"](state, action)
            else:
                assert len(action.shape) == 1, (
                    "Action should be specified as a 1-D tensor."
                    " However, shape of action was: " + str(action.shape)
                )
                assert action.shape[0] == self.action_space_dim, (
                    "Action shape is: "
                    + str(action.shape[0])
                    + ". Expected: "
                    + str(self.action_space_dim)
                )
                if self.action_space.contains(action):
                    # ### TODO implement for multiple orders, currently only for 1st order systems.
                    # if self.dynamics_order == 1:
                    #     next_state = state + action * self.time_unit / self.inertia

                    # print('self.state_derivatives:', self.state_derivatives)
                    # Except the last member of state_derivatives, the other occupy the same
                    # place in memory. Could create a new copy of them every time, but I think
                    # this should be more efficient and as long as tests don't fail should be
                    # fine.
                    # action is presumed to be n-th order force ##TODO Could easily scale this
                    # per dimension to give different kinds of dynamics per dimension: maybe
                    # even sample this scale per dimension from a probability distribution to
                    # generate different random Ps?
                    self.state_derivatives[-1] = (action / self.inertia)
                    factorial_array = scipy.special.factorial(
                        np.arange(1, self.dynamics_order + 1)
                    )  # This is just to speed things up as scipy calculates the factorial only for largest array member
                    for i in range(self.dynamics_order):
                        for j in range(self.dynamics_order - i):
                            # print('i, j, self.state_derivatives, (self.time_unit**(j + 1)), factorial_array:', i, j, self.state_derivatives, (self.time_unit**(j + 1)), factorial_array)
                            # + state_derivatives_prev[i] Don't need to add previous value as it's already in there at the beginning ##### TODO Keep an old self.state_derivatives and a new one otherwise higher order derivatives will be overwritten before being used by lower order ones.
                            self.state_derivatives[i] += (self.state_derivatives[i + j + 1] *
                                                          (self.time_unit ** (j + 1)) / factorial_array[j])
                    # print('self.state_derivatives:', self.state_derivatives)
                    next_state = self.state_derivatives[0]

                else:  # if action is from outside allowed action_space
                    next_state = state
                    warnings.warn(
                        "WARNING: Action "
                        + str(action)
                        + " out of range of action space. Applying 0 action!!"
                    )
            # if "transition_noise" in self.config:
            noise_in_transition = (
                self.transition_noise(self.np_random) if self.transition_noise else 0
            )  # #random
            self.total_abs_noise_in_transition_episode += np.abs(noise_in_transition)
            next_state += noise_in_transition  # ##IMP Noise is only applied to
            # state and not to higher order derivatives
            # TODO Check if next_state is within state space bounds
            if not self.observation_space.contains(next_state):
                self.logger.info(
                    "next_state out of bounds. next_state, clipping to"
                    + str(next_state)
                    + str(
                        np.clip(next_state, -self.state_space_max, self.state_space_max)
                    )
                )
                next_state = np.clip(
                    next_state, -self.state_space_max, self.state_space_max
                )
                # Could also "reflect"
                # next_state when it goes out of bounds. Would seem more logical
                # for a "wall", but would need to take care of multiple
                # reflections near a corner/edge.
                # Resets all higher order derivatives to 0
                zero_state = np.array([0.0] * (self.state_space_dim), dtype=self.dtype)
                # #####IMP to have copy() otherwise it's the same array
                # (in memory) at every position in the list:
                self.state_derivatives = [
                    zero_state.copy() for i in range(self.dynamics_order + 1)
                ]
                self.state_derivatives[0] = next_state

            if self.config["reward_function"] == "move_to_a_point":
                next_state_rel = np.array(next_state, dtype=self.dtype)[
                    self.config["relevant_indices"]
                ]
                dist_ = np.linalg.norm(next_state_rel - self.target_point)
                if dist_ < self.target_radius:
                    self.reached_terminal = True

        elif self.config["state_space_type"] == "grid":
            # state passed and returned is an np.array
            # Need to check that dtype is int because Gym doesn't
            if (
                self.action_space.contains(action)
                and np.array(action).dtype == np.int64
            ):
                if self.transition_noise:
                    # self.np_random.choice only works for 1-D arrays
                    if self.np_random.uniform() < self.transition_noise:  # #random
                        while True:  # Be careful of infinite loops
                            new_action = list(self.action_space.sample())  # #random
                            if new_action != action:
                                self.logger.info(
                                    "NOISE inserted! old action, new_action"
                                    + str(action)
                                    + str(new_action)
                                )
                                # print(str(action) + str(new_action))
                                self.total_noisy_transitions_episode += 1
                                action = new_action
                                break

                next_state = []
                for i in range(len(self.grid_shape)):
                    # actions -1, 0, 1 represent back, noop, forward respt.
                    next_state.append(state[i] + action[i])
                    if next_state[i] < 0:
                        self.logger.info("Underflow in grid next state. Bouncing back.")
                        next_state[i] = 0

                    if next_state[i] >= self.grid_shape[i]:
                        self.logger.info("Overflow in grid next state. Bouncing back.")
                        next_state[i] = self.grid_shape[i] - 1

            else:  # if action is from outside allowed action_space
                next_state = list(state)
                warnings.warn(
                    "WARNING: Action " + str(action) + " out of range"
                    " of action space. Applying noop action!!"
                )

            if self.config["reward_function"] == "move_to_a_point":
                if self.target_point == next_state:
                    self.reached_terminal = True

            next_state = np.array(next_state)

        return next_state

    def reward_function(self, state, action):
        """The reward function, R.

        Rewards the sequences selected to be rewardable at initialisation for discrete environments. For continuous environments, we have fixed available options for the reward function:
            move_to_a_point rewards for moving to a predefined location. It has sparse and dense settings.
            move_along_a_line rewards moving along ANY direction in space as long as it's a fixed direction for sequence_length consecutive steps.

        Parameters
        ----------
        state : list
            The underlying MDP state (also called augmented state in this code) that the environment uses to calculate its reward. Normally, just the sequence of past states of length delay + sequence_length + 1.
        action : single action dependent on action space
            Action magnitudes are penalised immediately in the case of continuous spaces and, in effect, play no role for discrete spaces as the reward in that case only depends on sequences of states. We say "in effect" because it _is_ used in case of a custom R to calculate R(s, a) but that is equivalent to using the "next" state s' as the reward determining criterion in case of deterministic transitions. _Sequences_ of _actions_ are currently NOT used to calculate the reward. Since the underlying MDP dynamics are deterministic, a state and action map 1-to-1 with the next state and so, just a sequence of _states_ should be enough to calculate the reward.

        Returns
        -------
        double
            The reward at the end of the current transition

        """

        # #TODO Make reward depend on the action sequence too instead of just state sequence, as it is currently?

        delay = self.delay
        sequence_length = self.sequence_length
        reward = 0.0
        # print("TEST", self.augmented_state[0 : self.augmented_state_length - delay], state, action, self.rewardable_sequences, type(state), type(self.rewardable_sequences))
        state_considered = state  # if imaginary_rollout else self.augmented_state # When we imagine a rollout, the user has to provide full augmented state as the argument!!
        # if not isinstance(state_considered, list):
        #     state_considered = [state_considered] # to get around case when sequence is an int; it should always be a list except if a user passes in a state; would rather force them to pass a list: assert for it!!
        # TODO These asserts are only needed if imaginary_rollout is True, as users then pass in a state sequence
        # if imaginary_rollout:
        #     assert isinstance(state_considered, list), "state passed in should be a list of states containing at the very least the state at beginning of the transition, s, and the one after it, s'. type was: " + str(type(state_considered))
        #     assert len(state_considered) == self.augmented_state_length, "Length of list of states passed should be equal to self.augmented_state_length. It was: " + str(len(state_considered))

        if self.use_custom_mdp:
            reward = self.config["reward_function"](state_considered, action)
            self.reward_buffer.append(reward)  # ##TODO Modify seq_len and delay
            # code for discrete and continuous case to use buffer too?
            reward = self.reward_buffer[0]
            # print("rewards:", self.reward_buffer, old_reward, reward)
            del self.reward_buffer[0]

        elif self.config["state_space_type"] == "discrete":
            if np.isnan(state_considered[0]):
                pass  # ###IMP: This check is to get around case of
                # augmented_state_length being > 2, i.e. non-vanilla seq_len or
                # delay, because then rewards may be handed out for the initial
                # state being part of a sequence which is not fair since it is
                # handed out without having the agent take an action.
            else:
                self.logger.debug(
                    "state_considered for reward:"
                    + str(state_considered)
                    + " with delay "
                    + str(self.delay)
                )
                if not self.reward_every_n_steps or (
                    self.reward_every_n_steps
                    and self.total_transitions_episode % self.sequence_length == delay
                ):
                    # ###TODO also implement this for make_denser case and continuous envs.
                    sub_seq = tuple(
                        state_considered[1: self.augmented_state_length - delay]
                    )
                    if sub_seq in self.rewardable_sequences:
                        # print(state_considered, "with delay", self.delay, "rewarded with:", 1)
                        reward += self.rewardable_sequences[sub_seq]
                    else:
                        # print(state_considered, "with delay", self.delay, "NOT rewarded.")
                        pass

                    self.logger.info("rew" + str(reward))

        elif self.config["state_space_type"] == "continuous":
            # ##TODO Make reward for along a line case to be length of line
            # travelled - sqrt(Sum of Squared distances from the line)? This
            # should help with keeping the mean reward near 0. Since the principal
            # component is always taken to be the direction of travel, this would
            # mean a larger distance covered in that direction and hence would
            # lead to +ve reward always and would mean larger random actions give
            # a larger reward! Should penalise actions in proportion that scale then?
            if np.isnan(state_considered[0][0]):  # Instead of below commented out
                # check, this is more robust for imaginary transitions
                # if self.total_transitions_episode + 1 < self.augmented_state_length:
                # + 1 because augmented_state_length is always 1 greater than seq_len + del
                pass  # #TODO
            else:
                if self.config["reward_function"] == "move_along_a_line":
                    # print("######reward test", self.total_transitions_episode, np.array(self.augmented_state), np.array(self.augmented_state).shape)
                    # #test: 1. for checking 0 distance for same action being always applied; 2. similar to 1. but for different dynamics orders; 3. similar to 1 but for different action_space_dims; 4. for a known applied action case, check manually the results of the formulae and see that programmatic results match: should also have a unit version of 4. for dist_of_pt_from_line() and an integration version here for total_deviation calc.?.
                    data_ = np.array(state_considered, dtype=self.dtype)[
                        1: self.augmented_state_length - delay,
                        self.config["relevant_indices"],
                    ]
                    data_mean = data_.mean(axis=0)
                    uu, dd, vv = np.linalg.svd(data_ - data_mean)
                    self.logger.info(
                        "uu.shape, dd.shape, vv.shape ="
                        + str(uu.shape)
                        + str(dd.shape)
                        + str(vv.shape)
                    )
                    line_end_pts = (
                        vv[0] * np.linspace(-1, 1, 2)[:, np.newaxis]
                    )  # vv[0] = 1st
                    # eigenvector, corres. to Principal Component #hardcoded -100
                    # to 100 to get a "long" line which should make calculations more
                    # robust(?: didn't seem to be the case for 1st few trials, so changed it
                    # to -1, 1; even tried up to 10000 - seems to get less precise for larger
                    # numbers) to numerical issues in dist_of_pt_from_line() below; newaxis
                    # added so that expected broadcasting takes place
                    line_end_pts += data_mean

                    total_deviation = 0
                    for (
                        data_pt
                    ) in (
                        data_
                    ):  # find total distance of all data points from the fit line above
                        total_deviation += dist_of_pt_from_line(
                            data_pt, line_end_pts[0], line_end_pts[-1]
                        )
                    self.logger.info(
                        "total_deviation of pts from fit line:" + str(total_deviation)
                    )

                    reward += -total_deviation / self.sequence_length

                elif self.config["reward_function"] == "move_to_a_point":  # Could
                    # generate target points randomly but leaving it to the user to do
                    # that. #TODO Generate it randomly to have random Rs?
                    if self.make_denser:
                        old_relevant_state = np.array(
                            state_considered, dtype=self.dtype
                        )[-2 - delay, self.config["relevant_indices"]]
                        new_relevant_state = np.array(
                            state_considered, dtype=self.dtype
                        )[-1 - delay, self.config["relevant_indices"]]
                        reward = -np.linalg.norm(new_relevant_state - self.target_point)
                        # Should allow other powers of the distance from target_point,
                        # or more norms?
                        reward += np.linalg.norm(old_relevant_state - self.target_point)
                        # Reward is the distance moved towards the target point.
                        # Should rather be the change in distance to target point, so reward given is +ve if "correct" action was taken and so reward function is more natural (this _is_ the current implementation)
                        # It's true that giving the total -ve distance from target as the loss at every step gives a stronger signal to algorithm to make it move faster towards target but this seems more natural (as in the other case loss/reward go up quadratically with distance from target point while in this case it's linear). The value function is in both cases higher for states further from target. But isn't that okay? Since the greater the challenge (i.e. distance from target), the greater is the achieved overall reward at the end.
                        # #TODO To enable seq_len, we can hand out reward if distance to target point is reduced (or increased - since that also gives a better signal than giving 0 in that case!!) for seq_len consecutive steps, otherwise 0 reward - however we need to hand out fixed reward for every "sequence" achieved otherwise, if we do it by adding the distance moved towards target in the sequence, it leads to much bigger rewards for larger seq_lens because of overlapping consecutive sequences.
                        # TODO also make_denser, sparse rewards only at target
                    else:  # sparse reward
                        new_relevant_state = np.array(
                            state_considered, dtype=self.dtype
                        )[-1 - delay, self.config["relevant_indices"]]
                        if (
                            np.linalg.norm(new_relevant_state - self.target_point)
                            < self.target_radius
                        ):
                            reward = 1.0  # Make the episode terminate as well?
                            # Don't need to. If algorithm is smart enough, it will
                            # stay in the radius and earn more reward.

                    reward -= self.action_loss_weight * np.linalg.norm(
                        np.array(action, dtype=self.dtype)
                    )

        elif self.config["state_space_type"] == "grid":
            if self.config["reward_function"] == "move_to_a_point":
                if self.make_denser:
                    old_relevant_state = np.array(state_considered)[-2 - delay]
                    new_relevant_state = np.array(state_considered)[-1 - delay]

                    manhat_dist_old = distance.cityblock(
                        old_relevant_state, np.array(self.target_point)
                    )
                    manhat_dist_new = distance.cityblock(
                        new_relevant_state, np.array(self.target_point)
                    )

                    reward += manhat_dist_old - manhat_dist_new

                else:  # sparse reward
                    new_relevant_state = np.array(state_considered)[-1 - delay]
                    if list(new_relevant_state) == self.target_point:
                        reward += 1.0

        reward *= self.reward_scale
        noise_in_reward = self.reward_noise(self.np_random) if self.reward_noise else 0
        # #random ###TODO Would be better to parameterise this in terms of state, action and time_step as well. Would need to change implementation to have a queue for the rewards achieved and then pick the reward that was generated delay timesteps ago.
        self.total_abs_noise_in_reward_episode += np.abs(noise_in_reward)
        self.total_reward_episode += reward
        reward += noise_in_reward
        reward += self.reward_shift
        return reward

    def step(self, action, imaginary_rollout=False):
        """The step function for the environment.

        Parameters
        ----------
        action : int or np.array
            The action that the environment will use to perform a transition.
        imaginary_rollout: boolean
            Option for the user to perform "imaginary" transitions, e.g., for model-based RL. If set to true, underlying augmented state of the MDP is not changed and user is responsible to maintain and provide a list of states to this function to be able to perform a rollout.

        Returns
        -------
        int or np.array, double, boolean, dict
            The next state, reward, whether the episode terminated and additional info dict at the end of the current transition
        """

        # For imaginary transitions, discussion:
        # 1) Use external observation_space as argument to P() and R(). But then it's not possible for P and R to know underlying MDP state unless we pass it as another argument. This is not desirable as we want P and R to simply be functions of external state/observation and action. 2) The other possibility is how it's currently done: P and R _know_ the underlying state. But in this case, we need an extra imaginary_rollout argument to P and R and we can't perform imaginary rollouts longer than one step without asking the user to maintain a sequence of underlying states and actions to be passed as arguments to P and R.
        # P and R knowing the underlying state seems a poor design choice to me
        # because it makes the code structure more brittle, so I propose that
        # step() handles the underlying state vs external observation conversion
        # and user can use P and R with underlying state. And step should handle
        # the case of imaginary rollouts by building a tree of transitions and
        # allowing rollback to states along the tree. However, user will probably
        # want access to P and R by using only observations as well instead of the
        # underlying state. In this case, P and R need to be aware of underlying
        # state and be able to store imaginary rollouts if needed.

        # Transform multi-discrete to discrete for discrete state spaces with
        # irrelevant dimensions; needed only for imaginary rollouts, otherwise,
        # internal augmented state is used.

        if imaginary_rollout:
            print("Imaginary rollouts are currently not supported.")
            sys.exit(1)

        if self.config["state_space_type"] == "discrete":
            if self.irrelevant_features:
                state, action, state_irrelevant, action_irrelevant = (
                    self.curr_state[0],
                    action[0],
                    self.curr_state[1],
                    action[1],
                )
            else:
                state, action = self.curr_state, action
        else:  # cont. or grid case
            state, action = self.curr_state, action

        # ### TODO Decide whether to give reward before or after transition ("after" would mean taking next state into account and seems more logical to me) - make it a dimension? - R(s) or R(s, a) or R(s, a, s')? I'd say give it after and store the old state in the augmented_state to be able to let the R have any of the above possible forms. That would also solve the problem of implicit 1-step delay with giving it before. _And_ would not give any reward for already being in a rewarding state in the 1st step but _would_ give a reward if 1 moved to a rewardable state - even if called with R(s, a) because s' is stored in the augmented_state! #####IMP

        # ###TODO P uses last state while R uses augmented state; for cont. env, P does know underlying state_derivatives - we don't want this to be the case for the imaginary rollout scenario;
        next_state = self.P(state, action)

        # if imaginary_rollout:
        #     pass
        #     # print("imaginary_rollout") # Since transition_function currently depends only on current state and action, we don't need to do anything here!
        # else:
        del self.augmented_state[0]
        if self.config["state_space_type"] == "discrete":
            self.augmented_state.append(next_state)
        elif self.config["state_space_type"] == "continuous":
            self.augmented_state.append(next_state.copy())
        elif self.config["state_space_type"] == "grid":
            self.augmented_state.append([next_state[i] for i in range(2)])

        self.total_transitions_episode += 1

        self.reward = self.R(self.augmented_state, action)

        # #irrelevant dimensions part
        if self.config["state_space_type"] == "discrete":
            if self.irrelevant_features:
                next_state_irrelevant = self.config["transition_function_irrelevant"][
                    state_irrelevant, action_irrelevant
                ]
                if self.transition_noise:
                    probs = (
                        np.ones(shape=(self.state_space_size[1],))
                        * self.transition_noise
                        / (self.state_space_size[1] - 1)
                    )
                    probs[next_state_irrelevant] = 1 - self.transition_noise
                    new_next_state_irrelevant = self.observation_spaces[1].sample(
                        prob=probs
                    )
                    # #random
                    # if next_state_irrelevant != new_next_state_irrelevant:
                    #     print("NOISE inserted! old next_state_irrelevant, new_next_state_irrelevant", next_state_irrelevant, new_next_state_irrelevant)
                    #     self.total_noisy_transitions_irrelevant_episode += 1
                    next_state_irrelevant = new_next_state_irrelevant

        # Transform discrete back to multi-discrete if needed
        if self.config["state_space_type"] == "discrete":
            if self.irrelevant_features:
                next_obs = next_state = (next_state, next_state_irrelevant)
            else:
                next_obs = next_state
        else:  # cont. or grid space
            next_obs = next_state

        if self.image_representations:
            next_obs = self.observation_space.get_concatenated_image(next_state)

        self.curr_state = next_state
        self.curr_obs = next_obs

        # #### TODO curr_state is external state, while we need to check relevant state for terminality! Done - by using augmented_state now instead of curr_state!
        self.done = (self.is_terminal_state(self.augmented_state[-1]) or self.reached_terminal)
        if self.done:
            self.reward += (
                self.term_state_reward * self.reward_scale
            )  # Scale before or after?
        self.logger.info(
            "sas'r:   "
            + str(self.augmented_state[-2])
            + "   "
            + str(action)
            + "   "
            + str(self.augmented_state[-1])
            + "   "
            + str(self.reward)
        )

        return self.curr_obs, self.reward, self.done, self.get_augmented_state()

    def get_augmented_state(self):
        """Intended to return the full augmented state which would be Markovian. (However, it's not Markovian wrt the noise in P and R because we're not returning the underlying RNG.) Currently, returns the augmented state which is the sequence of length "delay + sequence_length + 1" of past states for both discrete and continuous environments. Additonally, the current state derivatives are also returned for continuous environments.

        Returns
        -------
        dict
            Contains at the end of the current transition

        """
        # #TODO For noisy processes, this would need the noise distribution and random seed too. Also add the irrelevant state parts, etc.? We don't need the irrelevant parts for the state to be Markovian.
        if self.config["state_space_type"] == "discrete":
            augmented_state_dict = {
                "curr_state": self.curr_state,
                "curr_obs": self.curr_obs,
                "augmented_state": self.augmented_state,
            }
        elif self.config["state_space_type"] == "continuous":
            augmented_state_dict = {
                "curr_state": self.curr_state,
                "curr_obs": self.curr_obs,
                "augmented_state": self.augmented_state,
                "state_derivatives": self.state_derivatives,
            }
        elif self.config["state_space_type"] == "grid":
            augmented_state_dict = {
                "curr_state": self.curr_state,
                "curr_obs": self.curr_obs,
                "augmented_state": self.augmented_state,
            }

        return augmented_state_dict

    def reset(self):
        """Resets the environment for the beginning of an episode and samples a start state from rho_0. For discrete environments uses the defined rho_0 directly. For continuous environments, samples a state and resamples until a non-terminal state is sampled.

        Returns
        -------
        int or np.array
            The start state for a new episode.
        """

        # on episode "end" stuff (to not be invoked when reset() called when
        # self.total_episodes = 0; end is in quotes because it may not be a true
        # episode end reached by reaching a terminal state, but reset() may have
        # been called in the middle of an episode):
        if not self.total_episodes == 0:
            self.logger.info(
                "Noise stats for previous episode num.: "
                + str(self.total_episodes)
                + " (total abs. noise in rewards, total abs."
                " noise in transitions, total reward, total noisy transitions, total"
                " transitions): "
                + str(self.total_abs_noise_in_reward_episode)
                + " "
                + str(self.total_abs_noise_in_transition_episode)
                + " "
                + str(self.total_reward_episode)
                + " "
                + str(self.total_noisy_transitions_episode)
                + " "
                + str(self.total_transitions_episode)
            )

        # on episode start stuff:
        self.reward_buffer = [0.0] * (self.delay)

        self.total_episodes += 1

        if self.config["state_space_type"] == "discrete":
            self.curr_state_relevant = self.np_random.choice(
                self.state_space_size[0], p=self.config["relevant_init_state_dist"]
            )  # #random
            self.curr_state = self.curr_state_relevant  # # curr_state set here
            # already in case if statement below is not entered
            if self.irrelevant_features:
                self.curr_state_irrelevant = self.np_random.choice(
                    self.state_space_size[1],
                    p=self.config["irrelevant_init_state_dist"],
                )  # #random
                self.curr_state = (self.curr_state_relevant, self.curr_state_irrelevant)
                self.logger.info(
                    "RESET called. Relevant part of state reset to:"
                    + str(self.curr_state_relevant)
                )
                self.logger.info(
                    "Irrelevant part of state reset to:"
                    + str(self.curr_state_irrelevant)
                )

            self.augmented_state = [
                np.nan for i in range(self.augmented_state_length - 1)
            ]
            self.augmented_state.append(self.curr_state_relevant)
            # self.augmented_state = np.array(self.augmented_state) # Do NOT make an
            # np.array out of it because we want to test existence of the array in an
            # array of arrays which is not possible with np.array!
        elif self.config["state_space_type"] == "continuous":
            # self.logger.debug("#TODO for cont. spaces: reset")
            while True:  # Be careful about infinite loops
                term_space_was_sampled = False
                self.curr_state = self.feature_space.sample()  # #random
                if self.is_terminal_state(self.curr_state):
                    j = None
                    # Could this sampling be made more efficient? In general, the non-terminal
                    # space could have any shape and assiging equal sampling probability to
                    # each point in this space is pretty hard.
                    for i in range(len(self.term_spaces)):
                        if self.term_spaces[i].contains(self.curr_state):
                            j = i
                    self.logger.info(
                        "A state was sampled in term state subspace."
                        " Therefore, resampling. State was, subspace was:"
                        + str(self.curr_state)
                        + str(j)
                    )  # ##TODO Move this logic
                    # into a new class in Gym spaces that can contain
                    # subspaces for term states! (with warning/error if term
                    # subspaces cover whole state space, or even a lot of it)
                    term_space_was_sampled = True
                    # break
                if not term_space_was_sampled:
                    break

            # if not self.use_custom_mdp:
            # init the state derivatives needed for continuous spaces
            zero_state = np.array([0.0] * (self.state_space_dim), dtype=self.dtype)
            self.state_derivatives = [
                zero_state.copy() for i in range(self.dynamics_order + 1)
            ]  # #####IMP to have copy()
            # otherwise it's the same array (in memory) at every position in the list
            self.state_derivatives[0] = self.curr_state

            self.augmented_state = [
                [np.nan] * self.state_space_dim
                for i in range(self.augmented_state_length - 1)
            ]
            self.augmented_state.append(self.curr_state.copy())

        elif self.config["state_space_type"] == "grid":
            # Need to set self.curr_state, self.augmented_state
            while True:  # Be careful about infinite loops
                term_space_was_sampled = False
                # curr_state is an np.array while curr_state_relevant is a list
                self.curr_state = self.feature_space.sample().astype(int)  # #random
                self.curr_state_relevant = list(self.curr_state[[0, 1]])  # #hardcoded
                if self.is_terminal_state(self.curr_state_relevant):
                    self.logger.info(
                        "A terminal state was sampled. Therefore,"
                        " resampling. State was:" + str(self.curr_state)
                    )
                    term_space_was_sampled = True
                    break
                if not term_space_was_sampled:
                    break

            self.augmented_state = [
                np.nan for i in range(self.augmented_state_length - 1)
            ]
            self.augmented_state.append(self.curr_state_relevant)

        if self.image_representations:
            self.curr_obs = self.observation_space.get_concatenated_image(
                self.curr_state
            )
        else:
            self.curr_obs = self.curr_state

        self.logger.info("RESET called. curr_state reset to: " + str(self.curr_state))
        self.reached_terminal = False

        self.total_abs_noise_in_reward_episode = 0
        self.total_abs_noise_in_transition_episode = (
            0  # only present in continuous spaces
        )
        self.total_noisy_transitions_episode = 0  # only present in discrete spaces
        self.total_reward_episode = 0
        self.total_transitions_episode = 0

        self.logger.info(
            " self.delay, self.sequence_length:"
            + str(self.delay)
            + str(self.sequence_length)
        )

        return self.curr_obs

    def seed(self, seed=None):
        """Initialises the Numpy RNG for the environment by calling a utility for this in Gym.

        The environment has its own RNG and so do the state and action spaces held by the environment.

        Parameters
        ----------
        seed : int
            seed to initialise the np_random instance held by the environment. Cannot use numpy.int64 or similar because Gym doesn't accept it.

        Returns
        -------
        int
            The seed returned by Gym
        """
        # If seed is None, you get a randomly generated seed from gym.utils...
        self.np_random, self.seed_ = gym.utils.seeding.np_random(seed)  # #random
        print(
            "Env SEED set to: "
            + str(seed)
            + ". Returned seed from Gym: "
            + str(self.seed_)
        )
        return self.seed_


def dist_of_pt_from_line(pt, ptA, ptB):
    """Returns shortest distance of a point from a line defined by 2 points - ptA and ptB. Based on: https://softwareengineering.stackexchange.com/questions/168572/distance-from-point-to-n-dimensional-line"""

    tolerance = 1e-13
    lineAB = ptA - ptB
    lineApt = ptA - pt
    dot_product = np.dot(lineAB, lineApt)
    if np.linalg.norm(lineAB) < tolerance:
        return 0
    else:
        proj = dot_product / np.linalg.norm(
            lineAB
        )  # #### TODO could lead to division by zero if line is a null vector!
        sq_dist = np.linalg.norm(lineApt) ** 2 - proj ** 2

        if sq_dist < 0:
            if sq_dist < tolerance:
                logging.warning(
                    "The squared distance calculated in dist_of_pt_from_line()"
                    " using Pythagoras' theorem was less than the tolerance allowed."
                    " It was: " + str(sq_dist) + ". Tolerance was: -" + str(tolerance)
                )  # logging.warn() has been deprecated since Python
                # 3.3 and we should use logging.warning.
            sq_dist = 0
        dist = np.sqrt(sq_dist)
        #     print('pt, ptA, ptB, lineAB, lineApt, dot_product, proj, dist:', pt, ptA, ptB, lineAB, lineApt, dot_product, proj, dist)
        return dist


def list_to_float_np_array(lis):
    """Converts list to numpy float array"""
    return np.array(list(float(i) for i in lis))


if __name__ == "__main__":

    print("Please see example.py for how to use RLToyEnv.")
