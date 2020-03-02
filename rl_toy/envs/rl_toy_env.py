
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings
import numpy as np
import scipy
from scipy import stats
import gym
from gym.spaces import Discrete, BoxExtended, DiscreteExtended, MultiDiscreteExtended
#from gym.utils import seeding


class RLToyEnv(gym.Env):
    """Example of a custom env"""

    def __init__(self, config = None):
        """config can contain S, A, P, R, an initial distribution function over states (discrete or continuous), set of terminal states, gamma?,
        info to simulate a fixed/non-fixed delay in rewards (becomes non-Markovian then, need to keep a "delay" amount of previous states in memory),
        a non-determinism in state transitions, a non-determinism in rewards,


        To evaluate a learning algorithm we can compare its learnt models' KL-divergences with the true ones, or compare learnt Q and V with true ones.


        config["action_space_type"] = "discrete" or "continuous"
        config["state_space_type"] = "discrete" or "continuous"
        config["state_space_size"] = 6
        config["action_space_size"] = 6

        # config["state_space_dim"] = 2
        # config["action_space_dim"] = 1

        config["transition_function"]
        config["reward_function"]

        reward_range: A tuple corresponding to the min and max possible rewards?

        Do not want to write a new class with different parameters every time, so pass the desired functions in config! Could make a randomly generated function MDP class vs a fixed function one
        #tags # seed # hack # random # fix # TODO # IMP (more hashes means more important),
        #TODO Test cases to check all assumptions
        #TODO Mersenne Twister pseudo-random number generator used in Gym: not cryptographically secure!
        #TODO Different random seeds for S, A, P, R, rho_o and T? Currently 1 for env and 1 for its space.
        ###TODO Make sure terminal state is reachable by at least 1 path (no. of available actions affects avg. length of path from init. to term. state); no self transition loops?; Make sure each rewarded sequence is reachable; with equal S and A sizes we can make sure every state is reachable from every other state (just pick a random sequence to be the state tranition row for a state - solves previous problems sort of)
        Can check total no. of unique sequences of length, say, 3 reachable from diff. init. states based on the randomly generated transition function - could generate transition function and generate specific_sequences based on it.
        Maybe make some parts init states, some term. states and a majority be neither and let rewarded sequences begin from "neither" states (so we can see if algo. can learn to reach a reward-dense region and stay in it) - maybe keep only 1 init and 1 term. state?
        Can also make discrete/continuous "Euclidean space" with transition function fixed - then problem is with generating sequences (need to maybe from each init. state have a few length n rewardable sequences) - if distance from init state to term. state is large compared to sequence_length, exploration becomes easier? - you do get a signal if term. state is closer (like cliff walking problem) - still, things depend on percentage reachability of rewarded sequences (which increases if close term. states chop off a big part of the search space)
        #Check TODO, fix and bottleneck tags
        Sources of #randomness (Description needs to be updated for relevant/irrelevant update to playground): Seed for Env.observation_space (to generate discrete P, for noise in discrete P), Env.action_space (to generate initial random policy (done by the RL algorithm)), Env. (to generate R for discrete, for noise in R and continuous P, initial state), irrelevant state space?, irrelevant action space?, also for multi-discrete state and action spaces, self.term_spaces for continuous spaces - but multi-discrete state and term_spaces' sampling aren't used anywhere by Environment right now; ; Check # seed, # random
        ###TODO Separate out seeds for all the random processes? No, better to have one cryptographically secure PRNG I think! Or better yet, ask uuser to provide seeds for all processes, otherwise we can't get ALL possible distributions of random processes (state and action spaces, etc.) inside the Environment if we set all different seeds based on 1 seed because seeds for the processes inside will be set deterministically.
        ## TODO Terminal states: Gym expects _us_ to check for 'done' being true; rather set P(s, a) = s for any terminal state!
        ###IMP relevant_state_space_size should be large enough that after terminal state generation, we have enough num_specific_sequences rewardable!
        ###IMP Having irrelevant "dimensions" and performing transition dynamics in them for discrete spaces. I think we should not let the dynamics of the irrelevant "dimensions" interfere with the dynamics of the relevant "dimensions". This allows for a clean spearation of what we want the algorithm to pay attention to vs. what we don't it to pay attention to. For continuous spaces, we don't need to have a sperate observation_space with separate dynamics because the way the current dynamics work, the irrelevant and relevant dimensions are cleanly separated.
        ###IMP Variables I paid most attention to when adding irrelevant dimensions: config['state_space_size'], self.observation_space, self.curr_state. self.augmented_state
        #TODO Implement logger; command line argument parser; config from YAML file? No, YAML wouldn't allow programmatically modifiable config.
        #Discount factor Can be used by the algorithm if it wants it, more an intrinsic property of the algorithm. With delay, seq_len, etc., it's not clear how discount factor will work.
        #catastrophic forgetting: Could have multiple Envs as multiple tasks to tackle it.
        #intrinsic curiosity: Could have transition dynamics that have uncontrollable parts of the state space with the actions: could be done by just injecting noise for uncontrollable parts of the state space?
        """

        if config is None: # sets defaults
            config = {}

            # Discrete spaces configs:
            config["state_space_type"] = "discrete" # TODO if states are assumed categorical in discrete setting, need to have an embedding for their OHE when using NNs; do the encoding on the training end!
            config["action_space_type"] = "discrete"
            config["state_space_size"] = 6 # To be given as an integer for simple Discrete environment like Gym's. To be given as a list of integers for a MultiDiscrete environment like Gym's #TODO Rename state_space_size and action_space_size to be relevant_... wherever irrelevant dimensions are not used.
            config["action_space_size"] = 6

            # Continuous spaces configs:
            # config["state_space_type"] = "continuous"
            # config["action_space_type"] = "continuous"
            # config["state_space_dim"] = 2
            # config["action_space_dim"] = 2
            # config["transition_dynamics_order"] = 1
            # config["inertia"] = 1 # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
            # config["state_space_max"] = 5 # Will be a Box in the range [-max, max]
            # config["action_space_max"] = 5 # Will be a Box in the range [-max, max]
            # config["time_unit"] = 0.01 # Discretization of time domain
            config["terminal_states"] = [[0.0, 1.0], [1.0, 0.0]]
            config["term_state_edge"] =  1.0 # Terminal states will be in a hypercube centred around the terminal states given above with the edge of the hypercube of this length.

            # config for user specified P, R, rho_0, T. Examples here are for discrete spaces
            config["transition_function"] = np.array([[4 - i for i in range(config["state_space_size"])] for j in range(config["action_space_size"])]) #TODO ###IMP For all these prob. dist., there's currently a difference in what is returned for discrete vs continuous!
            config["reward_function"] = np.array([[4 - i for i in range(config["state_space_size"])] for j in range(config["action_space_size"])])
            config["init_state_dist"] = np.array([i/10 for i in range(config["state_space_size"])])
            config["is_terminal_state"] = np.array([config["state_space_size"] - 1]) # Can be discrete array or function to test terminal or not (e.g. for discrete and continuous spaces we may prefer 1 of the 2) #TODO currently always the same terminal state for a given environment state space size; have another variable named terminal_states to make semantic sense of variable name.


            config["generate_random_mdp"] = True ###IMP # This supersedes previous settings and generates a random transition function, a random reward function (for random specific sequences)
            config["delay"] = 0
            config["sequence_length"] = 3
            config["repeats_in_sequences"] = False
            config["reward_unit"] = 1.0
            config["reward_density"] = 0.25 # Number between 0 and 1
#            config["transition_noise"] = 0.2 # Currently the fractional chance of transitioning to one of the remaining states when given the deterministic transition function - in future allow this to be given as function; keep in mind that the transition function itself could be made a stochastic function - does that qualify as noise though?
#            config["reward_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to reward function
            # config["transition_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to transition function in cont. spaces
            config["make_denser"] = False
            config["terminal_state_density"] = 0.1 # Number between 0 and 1
            config["completely_connected"] = True # Make every state reachable from every state; If completely_connected, then no. of actions has to be at least equal to no. of states( - 1 if without self-loop); if repeating sequences allowed, then we have to have self-loops. Self-loops are ok even in non-repeating sequences - we just have a harder search problem then! Or make it maximally connected by having transitions to as many different states as possible - the row for P would have as many different transitions as possible!
            # print(config)
            #TODO asserts for the rest of the config settings
            # next: To implement delay, we can keep the previous observations to make state Markovian or keep an info bit in the state to denote that; Buffer length increase by fixed delay and fixed sequence length; current reward is incremented when any of the satisfying conditions (based on previous states) matches

            #seed old default settings used for paper, etc.:
            config["seed"] = {}
            config["seed"]["env"] = 0 # + config["seed"]
            config["seed"]["state_space"] = config["relevant_state_space_size"] # + config["seed"] for discrete, 10 + config["seed"] in case of continuous
            config["seed"]["action_space"] = 1 + config["relevant_action_space_size"] # + config["seed"] for discrete, 11 + config["seed"] in case of continuous
            # 12 + config["seed"] for continuous self.term_spaces

        if "seed" not in config:
            config["seed"] = None #seed
        if not isinstance(config["seed"], dict): # should be an int then. Gym doesn't accept np.int64, etc..
            seed_ = config["seed"]
            config["seed"] = {}
            config["seed"]["env"] = seed_
            self.seed(config["seed"]["env"]) #seed
            # All these diff. seeds may not be needed (you could have one seed for the joint relevant + irrelevant parts). But they allow for easy separation of the relevant and irrelevant dimensions!!
            config["seed"]["relevant_state_space"] = self.np_random.randint(sys.maxsize) #random
            config["seed"]["relevant_action_space"] = self.np_random.randint(sys.maxsize) #random
            config["seed"]["irrelevant_state_space"] = self.np_random.randint(sys.maxsize) #random
            config["seed"]["irrelevant_action_space"] = self.np_random.randint(sys.maxsize) #random
            config["seed"]["state_space"] = self.np_random.randint(sys.maxsize) #IMP This is currently used to sample only for continuous spaces and not used for discrete spaces by the Environment. User might want to sample from it for multi-discrete environments. #random
            config["seed"]["action_space"] = self.np_random.randint(sys.maxsize) #IMP This IS currently used to sample random actions by the RL agent for both discrete and continuous environments (but not used anywhere by the Environment). #random
        else: # if seed dict was passed
            self.seed(config["seed"]["env"]) #seed

        print('Seeds set to:', config["seed"])
        # print(f'Seeds set to {config["seed"]=}') # Available from Python 3.8


        #TODO Make below code more compact by reusing parts for state and action spaces?
        config["state_space_type"] = config["state_space_type"].lower()
        config["action_space_type"] = config["action_space_type"].lower()

        if "state_space_relevant_indices" not in config:
            config["state_space_relevant_indices"] = range(config["state_space_size"]) if config["state_space_type"] == "discrete" else range(config["state_space_dim"])
        else:
            pass

        if "action_space_relevant_indices" not in config:
            config["action_space_relevant_indices"] = range(config["action_space_size"]) if config["action_space_type"] == "discrete" else range(config["action_space_dim"])
        else:
            pass

        if ("init_state_dist" in config) and ("relevant_init_state_dist" not in config):
            config["relevant_init_state_dist"] = config["init_state_dist"]

        if config["state_space_type"] == "discrete":
            if isinstance(config["state_space_size"], list):
                config["state_space_multi_discrete_sizes"] = config["state_space_size"]
                self.relevant_state_space_maxes = np.array(config["state_space_size"])[np.array(config["state_space_relevant_indices"])]
                config["relevant_state_space_size"] = int(np.prod(self.relevant_state_space_maxes))
                config["state_space_irrelevant_indices"] = list(set(range(len(config["state_space_size"]))) - set(config["state_space_relevant_indices"]))
                if len(config["state_space_irrelevant_indices"]) == 0:
                    config["irrelevant_state_space_size"] = 0
                else:
                    self.irrelevant_state_space_maxes = np.array(config["state_space_size"])[np.array(config["state_space_irrelevant_indices"])]
                    # self.irrelevant_state_space_maxes = np.array(self.irrelevant_state_space_maxes)
                    config["irrelevant_state_space_size"] = int(np.prod(self.irrelevant_state_space_maxes))
            else: # if simple Discrete environment with the single "dimension" relevant
                assert type(config["state_space_size"]) == int, 'config["state_space_size"] has to be provided as an int when we have a simple Discrete environment. Was:' + str(type(config["state_space_size"]))
                config["relevant_state_space_size"] = config["state_space_size"]
                config["irrelevant_state_space_size"] = 0
            print('config["relevant_state_space_size"] inited to:', config["relevant_state_space_size"])
            print('config["irrelevant_state_space_size"] inited to:', config["irrelevant_state_space_size"])
        else: # if continuous environment
            pass

        if config["action_space_type"] == "discrete":
            if isinstance(config["action_space_size"], list):
                config["action_space_multi_discrete_sizes"] = config["action_space_size"]
                self.relevant_action_space_maxes = np.array(config["action_space_size"])[np.array(config["action_space_relevant_indices"])]
                config["relevant_action_space_size"] = int(np.prod(self.relevant_action_space_maxes))
                config["action_space_irrelevant_indices"] = list(set(range(len(config["action_space_size"]))) - set(config["action_space_relevant_indices"]))
                if len(config["action_space_irrelevant_indices"]) == 0:
                    config["irrelevant_action_space_size"] = 0
                else:
                    self.irrelevant_action_space_maxes = np.array(config["action_space_size"])[np.array(config["action_space_irrelevant_indices"])]
                    # self.irrelevant_action_space_maxes = np.array(self.irrelevant_action_space_maxes)
                    config["irrelevant_action_space_size"] = int(np.prod(self.irrelevant_action_space_maxes))
            else: # if simple Discrete environment with the single "dimension" relevant
                assert type(config["action_space_size"]) == int, 'config["action_space_size"] has to be provided as an int when we have a simple Discrete environment. Was:' + str(type(config["action_space_size"]))
                config["relevant_action_space_size"] = config["action_space_size"]
                config["irrelevant_action_space_size"] = 0
            print('config["relevant_action_space_size"] inited to:', config["relevant_action_space_size"])
            print('config["irrelevant_action_space_size"] inited to:', config["irrelevant_action_space_size"])
        else: # if continuous environment
            pass


        assert config["action_space_type"] == config["state_space_type"], 'config["state_space_type"] != config["action_space_type"]. Currently mixed space types are not supported.'
        assert config["sequence_length"] > 0, "config[\"sequence_length\"] <= 0. Set to: " + str(config["sequence_length"]) # also should be int
        if "completely_connected" in config and config["completely_connected"]:
            assert config["relevant_state_space_size"] == config["relevant_action_space_size"], "config[\"relevant_state_space_size\"] != config[\"relevant_action_space_size\"]. For completely_connected transition graphs, they should be equal. Please provide valid values. Vals: " + str(config["relevant_state_space_size"]) + " " + str(config["relevant_action_space_size"]) + ". In future, \"maximally_connected\" graphs are planned to be supported!"
            assert config["irrelevant_state_space_size"] == config["irrelevant_action_space_size"], "config[\"irrelevant_state_space_size\"] != config[\"irrelevant_action_space_size\"]. For completely_connected transition graphs, they should be equal. Please provide valid values! Vals: " + str(config["irrelevant_state_space_size"]) + " " + str(config["irrelevant_action_space_size"]) + ". In future, \"maximally_connected\" graphs are planned to be supported!" #TODO Currently, iirelevant dimensions have a P similar ot that of relevant dimensions. Should this be decoupled?

        self.config = config
        self.sequence_length = config["sequence_length"]
        self.delay = config["delay"]
        self.augmented_state_length = config["sequence_length"] + config["delay"]
        if self.config["state_space_type"] == "discrete":
            self.reward_unit = self.config["reward_unit"]
        else: # cont. spaces
            self.dynamics_order = self.config["transition_dynamics_order"]
            self.inertia = self.config["inertia"]
            self.time_unit = self.config["time_unit"]
            self.reward_scale = self.config["reward_scale"]

        self.total_episodes = 0

        # self.max_real = 100.0 # Take these settings from config
        # self.min_real = 0.0

        # def num_to_list(num1): #TODO Move this to a Utils file.
        #     if type(num1) == int or type(num1) == float:
        #         list1 = [num1]
        #     elif not isinstance(num1, list):
        #         raise TypeError("Argument to function should have been an int, float or list. Arg was: " + str(num1))
        #     return list1

        if config["state_space_type"] == "discrete":
            if config["irrelevant_state_space_size"] > 0:
                self.relevant_observation_space = DiscreteExtended(config["relevant_state_space_size"], seed=config["seed"]["relevant_state_space"]) #seed # hack #TODO Gym (and so Ray) apparently needs "observation"_space as a member. I'd prefer "state"_space
                self.irrelevant_observation_space = DiscreteExtended(config["irrelevant_state_space_size"], seed=config["seed"]["irrelevant_state_space"]) #seed # hack
                self.observation_space = MultiDiscreteExtended(config["state_space_size"], seed=config["seed"]["state_space"]) #seed # hack
            else:
                self.observation_space = DiscreteExtended(config["relevant_state_space_size"], seed=config["seed"]["relevant_state_space"]) #seed # hack
                self.relevant_observation_space = self.observation_space
                # print('id(self.observation_space)', id(self.observation_space), 'id(self.relevant_observation_space)', id(self.relevant_observation_space), id(self.relevant_observation_space) == id(self.observation_space))

        else:
            self.state_space_max = config["state_space_max"] if 'state_space_max' in config else np.inf # should we select a random max? #test?
            # config["state_space_max"] = num_to_list(config["state_space_max"]) * config["state_space_dim"]
            # print("config[\"state_space_max\"]", config["state_space_max"])

            self.observation_space = BoxExtended(-self.state_space_max, self.state_space_max, shape=(config["state_space_dim"], ), seed=config["seed"]["state_space"], dtype=np.float64) #seed #hack #TODO # low and high are 1st 2 and required arguments


        if config["action_space_type"] == "discrete":
            if config["irrelevant_state_space_size"] > 0:
                self.relevant_action_space = DiscreteExtended(config["relevant_action_space_size"], seed=config["seed"]["relevant_action_space"]) #seed # hack
                self.irrelevant_action_space = DiscreteExtended(config["irrelevant_action_space_size"], seed=config["seed"]["irrelevant_action_space"]) #seed # hack
                self.action_space = MultiDiscreteExtended(config["action_space_size"], seed=config["seed"]["action_space"]) #seed # hack
            else:
                self.action_space = DiscreteExtended(config["relevant_action_space_size"], seed=config["seed"]["relevant_action_space"]) #seed #hack #TODO
                self.relevant_action_space = self.action_space

        else:
            self.action_space_max = config["action_space_max"] if 'action_space_max' in config else np.inf #test?
            # config["action_space_max"] = num_to_list(config["action_space_max"]) * config["action_space_dim"]
            self.action_space = BoxExtended(-self.action_space_max, self.action_space_max, shape=(config["action_space_dim"], ), seed=config["seed"]["action_space"], dtype=np.float64) #seed # hack #TODO


        if not config["generate_random_mdp"]:
            #TODO When having a fixed delay/specific sequences, we need to have transition function from tuples of states of diff. lengths to next tuple of states. We can have this tupleness to have Markovianness on 1 or both of dynamics and reward functions.
            # P(s, a) = s' for deterministic transitions; or a probability dist. function over s' for non-deterministic #TODO keep it a probability function for deterministic too? pdf would be Dirac-Delta at specific points! Can be done in sympy but not scipy I think. Or overload function to return Prob dist.(s') function when called as P(s, a); or pdf or pmf(s, a, s') value when called as P(s, a, s')
            self.P = config["transition_function"] if callable(config["transition_function"]) else lambda s, a: config["transition_function"][s, a] # callable may not be optimal always since it was deprecated in Python 3.0 and 3.1
            # R(s, a) or (s, a , s') = r for deterministic rewards; or a probability dist. function over r for non-deterministic and then the return value of P() is a function, too! #TODO What happens when config is out of scope? Maybe use as self.config?
            self.R = config["reward_function"] if callable(config["reward_function"]) else lambda s, a: config["reward_function"][s, a]
            ##### TODO self.P and R were untended to be used as the dynamics functions inside step() - that's why were being inited by user-defined function here; but currently P is being used as dynamics for imagined transitions and transition_function is used for actual transitions in step() instead. So, setting P to manually configured transition means it won't be used in step() as was intended!
        else:
            #TODO Generate state and action space sizes also randomly?
            # Order of init is important!!
            self.init_terminal_states()
            self.init_init_state_dist()
            self.init_reward_function()
            self.init_transition_function()

        #TODO sample at any time from "current"/converged distribution of states according to current policy

        self.curr_state = self.reset() #TODO Maybe not call it here, since Gym seems to expect to _always_ call this method when using an environment; make this seedable? DO NOT do seed dependent initialization in reset() otherwise the initial state distrbution will always be at the same state at every call to reset()!! (Gym env has its own seed? Yes it does as also does space); extend Discrete, etc. spaces to sample states at init or at any time acc. to curr. policy?;
        print("self.augmented_state, len", self.augmented_state, len(self.augmented_state))

        print("toy env instantiated with config:", self.config) #hack

        # Reward: Have some randomly picked sequences that lead to rewards (can make it sparse or non-sparse setting). Sequence length depends on how difficult we want to make it.
        # print("self.P:", np.array([[self.P(i, j) for j in range(5)] for i in range(5)]), self.config["transition_function"])
        # print("self.R:", np.array([[self.R(i, j) for j in range(5)] for i in range(5)]), self.config["reward_function"])
        # self.R = self.reward_function
        # print("self.R:", self.R(0, 1))

    def init_terminal_states(self):
        if self.config["state_space_type"] == "discrete":
            self.num_terminal_states = int(self.config["terminal_state_density"] * self.config["relevant_state_space_size"])
            if self.num_terminal_states == 0: # Have at least 1 terminal state
                warnings.warn("WARNING: int(terminal_state_density * relevant_state_space_size) was 0. Setting num_terminal_states to be 1!")
                self.num_terminal_states = 1
            self.config["is_terminal_state"] = np.array([self.config["relevant_state_space_size"] - 1 - i for i in range(self.num_terminal_states)]) # terminal states inited to be at the "end" of the sorted states
            print("Inited terminal states to self.config['is_terminal_state']:", self.config["is_terminal_state"], "total", self.num_terminal_states)
            self.is_terminal_state = self.config["is_terminal_state"] if callable(self.config["is_terminal_state"]) else lambda s: s in self.config["is_terminal_state"]

        else: # if continuous space
            # print("#TODO for cont. spaces: term states")
            self.term_spaces = []
            # if 'terminal_states' not in self.config:
            #     self.config["terminal_states"] = []
            # if ('term_state_edge' not in self.config):
            #     self.config["term_state_edge"] = 0

            if 'terminal_states' in self.config: #test?
                for i in range(len(self.config["terminal_states"])):
                    lows = np.array([self.config["terminal_states"][i][j] - self.config["term_state_edge"]/2 for j in range(self.config["state_space_dim"])])
                    highs = np.array([self.config["terminal_states"][i][j] + self.config["term_state_edge"]/2 for j in range(self.config["state_space_dim"])])
                    # print("Term state lows, highs:", lows, highs)
                    self.term_spaces.append(BoxExtended(low=lows, high=highs, seed=12 + config["seed"], dtype=np.float64)) #seed #hack #TODO
                print("self.term_spaces samples:", self.term_spaces[0].sample(), self.term_spaces[-1].sample())

            self.is_terminal_state = lambda s: np.any([self.term_spaces[i].contains(s) for i in range(len(self.term_spaces))]) ### TODO for cont. #test?



    def init_init_state_dist(self):
        if self.config["state_space_type"] == "discrete":
            non_term_relevant_state_space_size = self.config["relevant_state_space_size"] - self.num_terminal_states
            self.config["relevant_init_state_dist"] = np.array([1 / (non_term_relevant_state_space_size) for i in range(non_term_relevant_state_space_size)] + [0 for i in range(self.num_terminal_states)]) #TODO Currently only uniform distribution over non-terminal states; Use Dirichlet distribution to select prob. distribution to use!
        #TODO make init_state_dist the default sample() for state space?
            # self.relevant_init_state_dist = self.config["relevant_init_state_dist"] if callable(self.config["relevant_init_state_dist"]) else lambda s: self.config["relevant_init_state_dist"][s] #TODO make the probs. sum to 1 by using Sympy/mpmath? self.relevant_init_state_dist is not used anywhere right now, self.config["relevant_init_state_dist"] is used!
            print("self.relevant_init_state_dist:", self.config["relevant_init_state_dist"])
        else: # if continuous space
            # print("#TODO for cont. spaces: init_state_dist")
            pass # this is handled in reset where we resample if we sample a term. state

        #irrelevant part
        if self.config["state_space_type"] == "discrete":
            if self.config["irrelevant_state_space_size"] > 0:
                self.config["irrelevant_init_state_dist"] = np.array([1 / (self.config["irrelevant_state_space_size"]) for i in range(self.config["irrelevant_state_space_size"])]) #TODO Currently only uniform distribution over non-terminal states; Use Dirichlet distribution to select prob. distribution to use!
                print("self.irrelevant_init_state_dist:", self.config["irrelevant_init_state_dist"])


    def init_reward_function(self):
        #TODO Maybe refactor this code and put useful reusable permutation generators, etc. in one library
        #print(self.config["reward_function"], "init_reward_function")
        if self.config["state_space_type"] == "discrete":
            non_term_relevant_state_space_size = self.config["relevant_state_space_size"] - self.num_terminal_states
            if self.config["repeats_in_sequences"]:
                num_possible_sequences = (self.relevant_observation_space.n - self.num_terminal_states) ** self.config["sequence_length"] #TODO if sequence cannot have replacement, use permutations; use state + action sequences? Subtracting the no. of terminal states from state_space size here to get "usable" states for sequences, having a terminal state even at end of reward sequence doesn't matter because to get reward we need to transition to next state which isn't possible for a terminal state.
                num_specific_sequences = int(self.config["reward_density"] * num_possible_sequences) #FIX Could be a memory problem if too large state space and too dense reward sequences
                self.specific_sequences = [[] for i in range(self.sequence_length)]
                sel_sequence_nums = self.np_random.choice(num_possible_sequences, size=num_specific_sequences, replace=False) #random # This assumes that all sequences have an equal likelihood of being selected for being a reward sequence;
                for i in range(num_specific_sequences):
                    curr_sequence_num = sel_sequence_nums[i]
                    specific_sequence = []
                    while len(specific_sequence) != self.config["sequence_length"]:
                        specific_sequence.append(curr_sequence_num % (non_term_relevant_state_space_size)) ####TODO
                        curr_sequence_num = curr_sequence_num // (non_term_relevant_state_space_size)
                    #bottleneck When we sample sequences here, it could get very slow if reward_density is high; alternative would be to assign numbers to sequences and then sample these numbers without replacement and take those sequences
                    # specific_sequence = self.relevant_observation_space.sample(size=self.config["sequence_length"], replace=True) # Be careful that sequence_length is less than state space size
                    self.specific_sequences[self.sequence_length - 1].append(specific_sequence) #hack
                    print("specific_sequence that will be rewarded", specific_sequence) #TODO impose a different distribution for these: independently sample state for each step of specific sequence; or conditionally dependent samples if we want something like DMPs/manifolds
                print("Total no. of rewarded sequences:", len(self.specific_sequences[self.sequence_length - 1]), "Out of", num_possible_sequences)
            else: # if no repeats_in_sequences
                len_ = self.sequence_length
                permutations = list(range(non_term_relevant_state_space_size + 1 - len_, non_term_relevant_state_space_size + 1))
                print("No. of choices for each element in a possible sequence (Total no. of permutations will be a product of this), 1 random number out of possible perms, no. of possible perms", permutations, np.random.randint(np.prod(permutations)), np.prod(permutations)) #random
                num_possible_permutations = np.prod(permutations)
                num_specific_sequences = int(self.config["reward_density"] * num_possible_permutations)
                if num_specific_sequences > 1000:
                    warnings.warn('Too many rewardable sequences and/or too long rewardable sequence length. Environment might be too slow. Please consider setting the reward_density to be lower or reducing the sequence length. No. of rewardable sequences:', num_specific_sequences) #TODO Maybe even exit the program if too much memory is (expected to be) taken.

                self.specific_sequences = [[] for i in range(self.sequence_length)]
                sel_sequence_nums = self.np_random.choice(num_possible_permutations, size=num_specific_sequences, replace=False) #random # This assumes that all sequences have an equal likelihood of being selected for being a reward sequence; # TODO this code could be replaced with self.np_random.permutation(non_term_relevant_state_space_size)[self.sequence_length]? Replacement becomes a problem then! We have to keep smpling until we have all unique rewardable sequences.
                total_clashes = 0
                for i in range(num_specific_sequences):
                    curr_permutation = sel_sequence_nums[i]
                    seq_ = []
                    curr_rem_digits = list(range(non_term_relevant_state_space_size)) # has to contain every number up to n so that any one of them can be picked as part of the sequence below
                    for j in permutations[::-1]: # Goes from largest to smallest number in nPk factors
                        rem_ = curr_permutation % j
                        seq_.append(curr_rem_digits[rem_])
                        del curr_rem_digits[rem_]
                #         print("curr_rem_digits", curr_rem_digits)
                        curr_permutation = curr_permutation // j
                #         print(rem_, curr_permutation, j, seq_)
                #     print("T/F:", seq_ in self.specific_sequences)
                    if seq_ in self.specific_sequences[self.sequence_length - 1]: #hack
                        total_clashes += 1 #TODO remove these extra checks and assert below
                    self.specific_sequences[self.sequence_length - 1].append(seq_)
                    print("specific_sequence that will be rewarded", seq_)
                #print(len(set(self.specific_sequences))) #error
                # print(self.specific_sequences[self.sequence_length - 1])

                print("Number of generated sequences that did not clash with an existing one when it was generated:", total_clashes)
                assert total_clashes == 0, 'None of the generated sequences should have clashed with an existing rewardable sequence when it was generated. No. of times a clash was detected:' + str(total_clashes)
                print("Total no. of rewarded sequences:", len(self.specific_sequences[self.sequence_length - 1]), "Out of", num_possible_permutations)
        else: # if continuous space
            print("#TODO for cont. spaces?: init_reward_function")

        self.R = lambda s, a: self.reward_function(s, a, only_query=False)


    def init_transition_function(self):

        # Future sequences don't depend on past sequences, only the next state depends on the past sequence of length, say n. n would depend on what order the dynamics are - 1st order would mean only 2 previous states needed to determine next state
        if self.config["state_space_type"] == "discrete":
            self.config["transition_function"] = np.zeros(shape=(self.config["relevant_state_space_size"], self.config["relevant_action_space_size"]), dtype=object)
            self.config["transition_function"][:] = -1 #IMP # To avoid having a valid value from the state space before we actually assign a usable value below!
            if self.config["completely_connected"]:
                for s in range(self.config["relevant_state_space_size"]):
                    self.config["transition_function"][s] = self.relevant_observation_space.sample(size=self.config["relevant_action_space_size"], replace=False) #random #TODO Preferably use the seed of the Env for this?
            else:
                for s in range(self.config["relevant_state_space_size"]):
                    for a in range(self.config["relevant_action_space_size"]):
                        self.config["transition_function"][s, a] = self.relevant_observation_space.sample() #random #TODO Preferably use the seed of the Env for this?
            for s in range(self.config["relevant_state_space_size"] - self.num_terminal_states, self.config["relevant_state_space_size"]):
                for a in range(self.config["relevant_action_space_size"]):
                    assert self.is_terminal_state(s) == True
                    self.config["transition_function"][s, a] = s # Setting P(s, a) = s for terminal states, for P() to be meaningful even if someone doesn't check for 'done' being = True

            print(self.config["transition_function"], "init_transition_function", type(self.config["transition_function"][0, 0]))
        else: # if continuous space
            print("#TODO for cont. spaces")


        #irrelevant part
        if self.config["state_space_type"] == "discrete":
            if self.config["irrelevant_state_space_size"] > 0: # What about irrelevant_ACTION_space_size > 0? Doesn't matter because only if an irrelevant state exists will an irrelevant action be used. #test to see if setting either irrelevant space to 0 causes crashes.
                self.config["transition_function_irrelevant"] = np.zeros(shape=(self.config["irrelevant_state_space_size"], self.config["irrelevant_action_space_size"]), dtype=object)
                self.config["transition_function_irrelevant"][:] = -1 #IMP # To avoid having a valid value from the state space before we actually assign a usable value below!
                if self.config["completely_connected"]:
                    for s in range(self.config["irrelevant_state_space_size"]):
                        self.config["transition_function_irrelevant"][s] = self.irrelevant_observation_space.sample(size=self.config["irrelevant_action_space_size"], replace=False) #random #TODO Preferably use the seed of the Env for this?
                else:
                    for s in range(self.config["irrelevant_state_space_size"]):
                        for a in range(self.config["irrelevant_action_space_size"]):
                            self.config["transition_function_irrelevant"][s, a] = self.irrelevant_observation_space.sample() #random #TODO Preferably use the seed of the Env for this?

                print(self.config["transition_function_irrelevant"], "init_transition_function _irrelevant", type(self.config["transition_function_irrelevant"][0, 0]))


        self.P = lambda s, a: self.transition_function(s, a, only_query = False)

    def reward_function(self, state, action, only_query=True): #TODO Make reward depend on state_action sequence instead of just state sequence? Maybe only use the action sequence for penalising action magnitude?
        # Transform multi-discrete to discrete if needed
        if self.config["state_space_type"] == "discrete":
            if isinstance(self.config["state_space_size"], list):
                if self.config["irrelevant_state_space_size"] == 0:
                    state, action, _, _ = self.multi_discrete_to_discrete(state, action)

        delay = self.delay
        sequence_length = self.sequence_length
        reward = 0.0
        # print("TEST", self.augmented_state[0 : self.augmented_state_length - delay], state, action, self.specific_sequences, type(state), type(self.specific_sequences))
        state_considered = state if only_query else self.augmented_state # When we imagine a rollout, the user has to provide full augmented state as the argument!!
        if not isinstance(state_considered, list):
            state_considered = [state_considered] # to get around case when sequence is an int

        if self.config["state_space_type"] == "discrete":
            if not self.config["make_denser"]:
                print(state_considered, "with delay", self.config["delay"])
                if state_considered[0 : self.augmented_state_length - delay] in self.specific_sequences[self.sequence_length - 1]:
                    # print(state_considered, "with delay", self.config["delay"], "rewarded with:", 1)
                    reward += self.reward_unit
                else:
                    # print(state_considered, "with delay", self.config["delay"], "NOT rewarded.")
                    pass
            else: # if make_denser
                for j in range(1, sequence_length + 1):
            # Check if augmented_states - delay up to different lengths in the past are present in sequence lists of that particular length; if so add them to the list of length
                    curr_seq_being_checked = state_considered[self.augmented_state_length - j - delay : self.augmented_state_length - delay]
                    # print("curr_seq_being_checked, self.possible_remaining_sequences[j - 1]:", curr_seq_being_checked, self.possible_remaining_sequences[j - 1])
                    if curr_seq_being_checked in self.possible_remaining_sequences[j - 1]:
                        count_ = self.possible_remaining_sequences[j - 1].count(curr_seq_being_checked)
                        # print("curr_seq_being_checked, count in possible_remaining_sequences, reward", curr_seq_being_checked, count_, count_ * self.reward_unit * j / self.sequence_length)
                        reward += count_ * self.reward_unit * j / self.sequence_length #TODO Maybe make it possible to choose not to multiply by count_ as a config option

                self.possible_remaining_sequences = [[] for i in range(sequence_length)] #TODO for variable sequence length just maintain a list of lists of lists rewarded_sequences
                for j in range(0, sequence_length):
            #        if j == 0:
                    for k in range(sequence_length):
                        for l in range(len(self.specific_sequences[k])): # self.specific_sequences[i][j][k] where 1st index is over different variable sequence lengths (to be able to support variable sequence lengths in the future), 2nd index is for the diff. sequences possible for that sequence length, 3rd index is over the sequence
                            if state_considered[self.augmented_state_length - j - delay : self.augmented_state_length - delay] == self.specific_sequences[k][l][:j]: # if curr_seq_being_checked matches a rewardable sequence up to a length j in the past, i.e., is a prefix of the rewardable sequence,
                                self.possible_remaining_sequences[j].append(self.specific_sequences[k][l][:j + 1]) # add it + an extra next state in that sequence to list of possible sequence prefixes to be checked for rewards above
                ###IMP: Above routine for sequence prefix checking can be coded in a more human understandable manner, but this kind of pruning out of sequences which may not be attainable based on the current past trajectory, done above, should be in principle more efficient?

                print("rew", reward)
                print("self.possible_remaining_sequences", self.possible_remaining_sequences)

        else: # if continuous space
            # print("#TODO for cont. spaces: noise")
            if self.total_transitions_episode + 1 < self.augmented_state_length: # + 1 because even without transition there may be reward as R() is before P() in step()
                pass #TODO
            else:
                # print("######reward test", self.total_transitions_episode, np.array(self.augmented_state), np.array(self.augmented_state).shape)
                #test: 1. for checking 0 distance for same action being always applied; 2. similar to 1. but for different dynamics orders; 3. similar to 1 but for different action_space_dims; 4. for a known applied action case, check manually the results of the formulae and see that programmatic results match: should also have a unit version of 4. for dist_of_pt_from_line() and an integration version here for total_dist calc.?.
                data_ = np.array(self.augmented_state)[0 : self.augmented_state_length - delay, :]
                data_mean = data_.mean(axis=0)
                uu, dd, vv = np.linalg.svd(data_ - data_mean)
                print('uu.shape, dd.shape, vv.shape =', uu.shape, dd.shape, vv.shape)
                line_end_pts = vv[0] * np.linspace(-1, 1, 2)[:, np.newaxis] # vv[0] = 1st eigenvector, corres. to Principal Component #hardcoded -100 to 100 to get a "long" line which should make calculations more robust(?: didn't seem to be the case for 1st few trials, so changed it to -1, 1; even tried up to 10000- seems to get less precise for larger numbers) to numerical issues in dist_of_pt_from_line() below; newaxis added so that expected broadcasting takes place
                line_end_pts += data_mean

                total_dist = 0
                for data_pt in data_: # find total distance of all data points from the fit line above
                    total_dist += dist_of_pt_from_line(data_pt, line_end_pts[0], line_end_pts[-1])
                print('total_dist of pts from fit line:', total_dist)

                reward += ( - total_dist / self.sequence_length ) * self.reward_scale

                # x = np.array(self.augmented_state)[0 : self.augmented_state_length - delay, 0]
                # y = np.array(self.augmented_state)[0 : self.augmented_state_length - delay, 1]
                # A = np.vstack([x, np.ones(len(x))]).T
                # coeffs, sum_se, rank_A, singular_vals_A = np.linalg.lstsq(A, y, rcond=None)
                # sum_se = sum_se[0]
                # reward += (- np.sqrt(sum_se / self.sequence_length)) * self.reward_scale

                # slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                # reward += (1 - std_err) * self.reward_scale

                # print("rew, rew / seq_len", reward, reward / self.sequence_length)
                # print(slope, intercept, r_value, p_value, std_err)


        noise_in_reward = self.config["reward_noise"](self.np_random) if "reward_noise" in self.config else 0 #random
        self.total_abs_noise_in_reward_episode += np.abs(noise_in_reward)
        self.total_reward_episode += reward
        reward += noise_in_reward
        return reward

    def transition_function(self, state, action, only_query=True):
        # only_query when true performs an imaginary transition i.e. augmented state etc. are not updated;
        # print("Before transition", self.augmented_state)

        # Transform multi-discrete to discrete if needed
        if self.config["state_space_type"] == "discrete":
            if isinstance(self.config["state_space_size"], list):
                if self.config["irrelevant_state_space_size"] > 0:
                    state, action, state_irrelevant, action_irrelevant = self.multi_discrete_to_discrete(state, action, irrelevant_parts=True)
                else:
                    state, action, _, _ = self.multi_discrete_to_discrete(state, action)

        if self.config["state_space_type"] == "discrete":
            next_state = self.config["transition_function"][state, action]
            if "transition_noise" in self.config:
                probs = np.ones(shape=(self.config["relevant_state_space_size"],)) * self.config["transition_noise"] / (self.config["relevant_state_space_size"] - 1)
                probs[next_state] = 1 - self.config["transition_noise"]
                # TODO Samples according to new probs to get noisy discrete transition
                new_next_state = self.relevant_observation_space.sample(prob=probs) #random
                # print("noisy old next_state, new_next_state", next_state, new_next_state)
                if next_state != new_next_state:
                    print("NOISE inserted! old next_state, new_next_state", next_state, new_next_state)
                    self.total_noisy_transitions_episode += 1
                # print("new probs:", probs, self.relevant_observation_space.sample(prob=probs))
                next_state = new_next_state
                # assert np.sum(probs) == 1, str(np.sum(probs)) + " is not equal to " + str(1)

            #irrelevant part
            if self.config["irrelevant_state_space_size"] > 0:
                if self.config["irrelevant_action_space_size"] > 0: # only if there's an irrelevant action does a transition take place (even a noisy one)
                    next_state_irrelevant = self.config["transition_function_irrelevant"][state_irrelevant, action_irrelevant]
                    if "transition_noise" in self.config:
                        probs = np.ones(shape=(self.config["irrelevant_state_space_size"],)) * self.config["transition_noise"] / (self.config["irrelevant_state_space_size"] - 1)
                        probs[next_state_irrelevant] = 1 - self.config["transition_noise"]
                        new_next_state_irrelevant = self.irrelevant_observation_space.sample(prob=probs) #random
                        # if next_state_irrelevant != new_next_state_irrelevant:
                        #     print("NOISE inserted! old next_state_irrelevant, new_next_state_irrelevant", next_state_irrelevant, new_next_state_irrelevant)
                        #     self.total_noisy_transitions_irrelevant_episode += 1
                        next_state_irrelevant = new_next_state_irrelevant


        else: # if continuous space
            # print("#TODO for cont. spaces: noise")
            assert len(action.shape) == 1, 'Action should be specified as a 1-D tensor. However, shape of action was: ' + str(action.shape)
            assert action.shape[0] == self.config['action_space_dim'], 'Action shape is: ' + str(action.shape[0]) + '. Expected: ' + str(self.config['action_space_dim'])
            if self.action_space.contains(action):
                ### TODO implement for multiple orders, currently only for 1st order systems.
                # if self.dynamics_order == 1:
                #     next_state = state + action * self.time_unit / self.inertia

                print('self.state_derivatives:', self.state_derivatives)
                # Except the last member of state_derivatives, the other occupy the same place in memory. Could create a new copy of them every time, but I think this should be more efficient and as long as tests don't fail should be fine.
                self.state_derivatives[-1] = action / self.inertia # action is presumed to be n-th order force
                factorial_array = scipy.special.factorial(np.arange(1, self.config['transition_dynamics_order'] + 1)) # This is just to speed things up as scipy calculates the factorial only for largest array member
                for i in range(self.config['transition_dynamics_order']):
                    for j in range(self.config['transition_dynamics_order'] - i):
                        print(i, j, self.state_derivatives, (self.time_unit**(j + 1)), factorial_array)
                        self.state_derivatives[i] += self.state_derivatives[i + j + 1] * (self.time_unit**(j + 1)) / factorial_array[j] #+ state_derivatives_prev[i] Don't need to add previous value as it's already in there at the beginning ##### TODO Keep an old self.state_derivatives and a new one otherwise higher order derivatives will be overwritten before being used by lower order ones.
                print('self.state_derivatives:', self.state_derivatives)
                next_state = self.state_derivatives[0]

            else: # if action is from outside allowed action_space
                next_state = state
                warnings.warn("WARNING: Action out of range of action space. Applying 0 action!!")
            # if "transition_noise" in self.config:
            noise_in_transition = self.config["transition_noise"](self.np_random) if "transition_noise" in self.config else 0 #random
            self.total_abs_noise_in_transition_episode += np.abs(noise_in_transition)
            next_state += noise_in_transition
            ### TODO Check if next_state is within state space bounds
            if not self.observation_space.contains(next_state):
                print("next_state out of bounds. next_state, clipping to", next_state, np.clip(next_state, -self.state_space_max, self.state_space_max))
                next_state = np.clip(next_state, -self.state_space_max, self.state_space_max) # Could also "reflect" next_state when it goes out of bounds. Would seem more logical for a "wall", but need to take care of multiple reflections near a corner/edge.

        if only_query:
            pass
            # print("Only query") # Since transition_function currently depends only on current state and action, we don't need to do anything here!
        else:
            del self.augmented_state[0]
            self.augmented_state.append(next_state)
            self.total_transitions_episode += 1


        # Transform discrete back to multi-discrete if needed
        if self.config["state_space_type"] == "discrete":
            if isinstance(self.config["state_space_size"], list):
                if self.config["irrelevant_state_space_size"] > 0:
                    next_state = self.discrete_to_multi_discrete(next_state, next_state_irrelevant)
                else:
                    next_state = self.discrete_to_multi_discrete(next_state)
            # print("After transition", self.augmented_state)
        #TODO Check ergodicity of MDP/policy? Can calculate probability of a rewardable specific sequence occurring (for a random policy)
        # print("TEEEEEST:", self.config["transition_function"], [state, action])
        # if next_state in self.config["is_terminal_state"]:
        #     print("NEXT_STATE:", next_state, next_state in self.config["is_terminal_state"])
        return next_state

    def discrete_to_multi_discrete(self, relevant_part, irrelevant_part=None):
        '''
        Transforms relevant and irrelevant parts of state (NOT action) space from discrete to its multi-discrete representation which is the externally visible observation_space from the Environment
        #TODO Generalise function to also be able to transform actions
        '''
        relevant_part = transform_discrete_to_multi_discrete(relevant_part, self.relevant_state_space_maxes)
        combined_ = relevant_part
        if self.config["irrelevant_state_space_size"] > 0:
            irrelevant_part = transform_discrete_to_multi_discrete(irrelevant_part, self.irrelevant_state_space_maxes)

            combined_ = np.zeros(shape=(len(self.config['state_space_size']),), dtype=int)
            combined_[self.config["state_space_relevant_indices"]] = relevant_part
            combined_[self.config["state_space_irrelevant_indices"]] = irrelevant_part

        return list(combined_)

    def multi_discrete_to_discrete(self, state, action, irrelevant_parts=False):
        '''
        Transforms multi-discrete representations of state and action to their discrete equivalents. Needed at the beginnings of P and R to convert externally visible observation_space from the Environment to the internal observation_space that is used inside P and R
        '''
        relevant_part_state = transform_multi_discrete_to_discrete(np.array(state)[np.array(self.config['state_space_relevant_indices'])], self.relevant_state_space_maxes)
        relevant_part_action = transform_multi_discrete_to_discrete(np.array(action)[np.array(self.config['action_space_relevant_indices'])], self.relevant_action_space_maxes)
        irrelevant_part_state = None
        irrelevant_part_action = None

        if irrelevant_parts:
            irrelevant_part_state = transform_multi_discrete_to_discrete(np.array(state)[np.array(self.config['state_space_irrelevant_indices'])], self.irrelevant_state_space_maxes)
            irrelevant_part_action = transform_multi_discrete_to_discrete(np.array(action)[np.array(self.config['action_space_irrelevant_indices'])], self.irrelevant_action_space_maxes)

        return relevant_part_state, relevant_part_action, irrelevant_part_state, irrelevant_part_action

    def get_augmented_state(self):
        '''
        Intended to return the full augmented state which would be Markovian. For noisy processes, this would need the noise distribution and random seed too? Also add the irrelevant state parts, etc.? #TODO
        '''
        return {"curr_state": self.curr_state, "augmented_state": self.augmented_state}

    def reset(self):
        # TODO reset is also returning info dict to be able to return state in addition to observation;
        # TODO Do not start in a terminal state.

        # on episode "end" stuff (to not be invoked when reset() called when self.total_episodes = 0; end is quoted because it may not be a true episode end reached by reaching a terminal state, but reset() may have been called in the middle of an episode):
        if not self.total_episodes == 0:
            print("Noise stats for previous episode num.:", self.total_episodes, "(total abs. noise in rewards, total abs. noise in transitions, total reward, total noisy transitions, total transitions):",
                    self.total_abs_noise_in_reward_episode, self.total_abs_noise_in_transition_episode, self.total_reward_episode, self.total_noisy_transitions_episode,
                    self.total_transitions_episode)

        # on episode start stuff:
        self.total_episodes += 1

        if self.config["state_space_type"] == "discrete":
            self.curr_state_relevant = self.np_random.choice(self.config["relevant_state_space_size"], p=self.config["relevant_init_state_dist"]) #random
            self.curr_state = self.curr_state_relevant
            if isinstance(self.config["state_space_size"], list):
                if self.config["irrelevant_state_space_size"] > 0:
                    self.curr_state_irrelevant = self.np_random.choice(self.config["irrelevant_state_space_size"], p=self.config["irrelevant_init_state_dist"]) #random
                    self.curr_state = self.discrete_to_multi_discrete(self.curr_state_relevant, self.curr_state_irrelevant)
                    print("RESET called. Relevant part of state reset to:", self.curr_state_relevant)
                    print("Irrelevant part of state reset to:", self.curr_state_irrelevant)
                else:
                    self.curr_state = self.discrete_to_multi_discrete(self.curr_state_relevant)
                    print("RESET called. Relevant part of state reset to:", self.curr_state_relevant)

            print("RESET called. curr_state set to:", self.curr_state)
            self.augmented_state = [np.nan for i in range(self.augmented_state_length - 1)]
            self.augmented_state.append(self.curr_state_relevant)
            # self.augmented_state = np.array(self.augmented_state) # Do NOT make an np.array out of it because we want to test existence of the array in an array of arrays
        else: # if continuous space
            print("#TODO for cont. spaces: reset")
            while True: # Be careful about infinite loops
                term_space_was_sampled = False
                self.curr_state = self.observation_space.sample() #random
                for i in range(len(self.term_spaces)): # Could this sampling be made more efficient? In general, the non-terminal space could have any shape and assiging equal sampling probability to each point in this space is pretty hard.
                    if self.term_spaces[i].contains(self.curr_state):
                        print("A state was sampled in term state subspace. Therefore, resampling. State was, subspace was:", self.curr_state, i) ##TODO Move this logic into a new class in Gym spaces that can contain subspaces for term states! (with warning/error if term subspaces cover whole state space, or even a lot of it)
                        term_space_was_sampled = True
                        break
                if not term_space_was_sampled:
                    break

            # init the state derivatives needed for continuous spaces
            zero_state = np.array([0.0] * (self.config['state_space_dim']))
            self.state_derivatives = [zero_state.copy() for i in range(self.config['transition_dynamics_order'] + 1)] #####IMP to have copy() otherwise it's the same array (in memory) at every position in the list
            self.state_derivatives[0] = self.curr_state

            print("RESET called. State reset to:", self.curr_state)
            self.augmented_state = [[np.nan] * self.config["state_space_dim"] for i in range(self.augmented_state_length - 1)]
            self.augmented_state.append(self.curr_state)

        self.total_abs_noise_in_reward_episode = 0
        self.total_abs_noise_in_transition_episode = 0 # only present in continuous spaces
        self.total_noisy_transitions_episode = 0 # only present in discrete spaces
        self.total_reward_episode = 0
        self.total_transitions_episode = 0


        # This part just initializes self.possible_remaining_sequences to hold 1st state in all rewardable sequences, which will be checked for after 1st step of the episode to give rewards.
        ### TODO Move this part to reset()? Done. But even before, this small bug shouldn't have caused a problem because the sub-sequences, i.e. prefixes, of length >1 in possible_remaining_sequences would have been checked against NaNs for the past states and would have not contributed to the reward.
        if self.config["state_space_type"] == "discrete" and self.config["make_denser"] == True:
            delay = self.delay
            sequence_length = self.sequence_length
            self.possible_remaining_sequences = [[] for i in range(sequence_length)]
            for j in range(1):
            #        if j == 0:
                for k in range(sequence_length):
                    for l in range(len(self.specific_sequences[k])):
    #                    if state_considered[self.augmented_state_length - j - delay : self.augmented_state_length - delay] == self.specific_sequences[k][l][:j]:
                            self.possible_remaining_sequences[j].append(self.specific_sequences[k][l][:j + 1])

            print("self.possible_remaining_sequences", self.possible_remaining_sequences)
            print(" self.delay, self.sequence_length:", self.delay, self.sequence_length)

        return self.curr_state

    def step(self, action):
        # assert self.action_space.contains(action) , str(action) + " not in" # TODO Need to implement check in for this environment
        #TODO check self.done and throw error if it's True? (Gym doesn't do it!); Otherwise, define self transitions in terminal states
        self.reward = self.R(self.curr_state, action) #TODO Decide whether to give reward before or after transition ("after" would mean taking next state into account and seems more logical to me)
        self.curr_state = self.P(self.curr_state, action)


        self.done = self.is_terminal_state(self.curr_state)
        return self.curr_state, self.reward, self.done, {"curr_state": self.curr_state}

    def seed(self, seed=None):
        # If seed is None, you get a randomly generated seed from gym.utils...
        # if seed is not None:
        self.np_random, self.seed_ = gym.utils.seeding.np_random(seed) #random
        print("Env SEED set to:", seed, "Returned seed from Gym:", self.seed_)
        return self.seed_



def transform_multi_discrete_to_discrete(vector, vector_maxes):
    '''
    Transforms a multi-discrete vector drawn from a multi-discrete space with ranges for each dimension from 0 to vector_maxes to a discrete equivalent with the discrete number drawn from a 1-D space where the min is 0 and the max is np.prod(vector_maxes) - 1. The correspondence between "counting"/ordering in the multi-discrete space with the discrete space assumes that the rightmost element varies most frequently in the multi-discrete space.
    '''
    return np.arange(np.prod(vector_maxes)).reshape(vector_maxes)[tuple(vector)]

def transform_discrete_to_multi_discrete(scalar, vector_maxes):
    '''
    Transforms a discrete scalar drawn from a 1-D space where the min is 0 and the max is np.prod(vector_maxes) - 1 to a multi-discrete equivalent with the multi-discrete vector drawn from a multi-discrete space with ranges for each dimension from 0 to vector_maxes
    '''
    return np.argwhere(np.arange(np.prod(vector_maxes)).reshape(vector_maxes) == scalar).flatten()


# class OT4RL():
#     """Functions to perform calculations of metrics like Wasserstein distance. Useful for tracking learning progress by getting the distance between learnt and true models."""
#
#     def __init__(self, config = None):

def mean_sinkhorn_dist(model_1_P, model_2_P, model_1_R=None, model_2_R=None, weighted=True):
    """Calculates the mean approx. Wasserstein dist. between P(s, a) distributions over all the possible (s, a) values, each dist. weighted by the |R(s, a)| magnitudes. Can optionally be unweighted.
       If model_1_R=None and model_2_R=None, then model_1_P and model_2_P can be general models for which averaged approx. Wasserstein dist. (i.e. sinkhorn divergence) is calculated, for e.g., they could even be models of R().
    """

#distance of a point from a line: https://softwareengineering.stackexchange.com/questions/168572/distance-from-point-to-n-dimensional-line
def dist_of_pt_from_line(pt, ptA, ptB):
    lineAB = ptA - ptB
    lineApt = ptA - pt
    dot_product = np.dot(lineAB, lineApt)
    proj = dot_product / np.linalg.norm(lineAB) ####TODO could lead to division by zero if line is a null vector!
    sq_dist = np.linalg.norm(lineApt)**2 - proj**2
    tolerance = -1e-13
    if sq_dist < 0:
        if sq_dist < tolerance:
            warnings.warn('The squared distance calculated in dist_of_pt_from_line() using Pythagoras\' theorem was less than the tolerance allowed. It was: ' + str(sq_dist) + '. Tolerance was: ' + str(tolerance))
        sq_dist = 0
    dist = np.sqrt(sq_dist)
#     print('pt, ptA, ptB, lineAB, lineApt, dot_product, proj, dist:', pt, ptA, ptB, lineAB, lineApt, dot_product, proj, dist)
    return dist

if __name__ == "__main__":

    config = {}
    config["seed"] = 0 #seed, 7 worked for initially sampling within term state subspace

    # config["state_space_type"] = "discrete"
    # config["action_space_type"] = "discrete"
    # config["relevant_state_space_size"] = 6
    # config["relevant_action_space_size"] = 6
    # config["reward_density"] = 0.25 # Number between 0 and 1
    # config["make_denser"] = True
    # config["terminal_state_density"] = 0.25 # Number between 0 and 1
    # config["completely_connected"] = True # Make every state reachable from every state
    # config["repeats_in_sequences"] = False
    # config["delay"] = 1
    # config["sequence_length"] = 3
    # config["reward_unit"] = 1.0


    config["state_space_type"] = "continuous"
    config["action_space_type"] = "continuous"
    config["state_space_dim"] = 4
    config["action_space_dim"] = 4
    config["transition_dynamics_order"] = 1
    config["inertia"] = 1 # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
    # config["state_space_max"] = 5 # Will be a Box in the range [-max, max]
    # config["action_space_max"] = 1 # Will be a Box in the range [-max, max]
    config["time_unit"] = 1 # Discretization of time domain
    # config["terminal_states"] = [[0.0, 1.0], [1.0, 0.0]]
    # config["term_state_edge"] =  1.0 # Terminal states will be in a hypercube centred around the terminal states given above with the edge of the hypercube of this length.

    config["delay"] = 1
    config["sequence_length"] = 10
    config["reward_scale"] = 1.0
#    config["transition_noise"] = 0.2 # Currently the fractional chance of transitioning to one of the remaining states when given the deterministic transition function - in future allow this to be given as function; keep in mind that the transition function itself could be made a stochastic function - does that qualify as noise though?
    # config["reward_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to reward function
    # config["transition_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to transition function in cont. spaces

    config["generate_random_mdp"] = True # This supersedes previous settings and generates a random transition function, a random reward function (for random specific sequences)
    env = RLToyEnv(config)
#    env.seed(0)
#    from rl_toy.envs import RLToyEnv
#    env = gym.make("RLToy-v0")
#    print("env.spec.max_episode_steps, env.unwrapped:", env.spec.max_episode_steps, env.unwrapped)
    state = env.get_augmented_state()['curr_state']
    # print("TEST", type(state))
    for _ in range(20):
        # env.render() # For GUI
        action = env.action_space.sample() # take a #random action # TODO currently DiscreteExtended returns a sampled array
        # action = np.array([1, 1, 1, 1]) # just to test if acting "in a line" works
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        state = next_state
    env.reset()
    env.close()

    # import sys
    # sys.exit(0)
