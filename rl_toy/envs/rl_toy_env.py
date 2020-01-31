
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
import gym
from gym.spaces import Discrete, BoxExtended, DiscreteExtended
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
        Can also make discrete/continuous "Euclidean space" with transition function fixed - then problem is with generating sequences (need to maybe from each init. state have a few length n rewardable sequences) - if distance from init state to term. state is large compared to sequence_length, exploration becomes easier? - you do get a signal is term. state is closer (like cliff walking problem) - still, things depend on percentage reachability of rewarded sequences (which increases if close term. states chop off a big part of the search space)
        #Check TODO, fix and bottleneck tags
        Sources of #randomness: Seed for Env.observation_space (to generate P, for noise in P), Env.action_space (to generate initial random policy (done by the RL algorithm)), Env. (to generate R, for noise in R, initial state); ; Check # seed, # random
        ###TODO Separate out seeds for all the random processes!
        ###IMP state_space_size should be large enough that after terminal state generation, we have enough num_specific_sequences rewardable!
        #TODO Implement logger, command line argument parser, config from YAML file?
        #Discount factor Can be used by the algorithm if it wants it, more an intrinsic property of the algorithm. With delay, seq_len, etc., it's not clear how discount factor will work.
        """

        if config is None:
            config = {}

            # Discrete spaces configs:
            config["state_space_type"] = "discrete" # TODO if states are assumed categorical in discrete setting, need to have an embedding for their OHE when using NNs; do the encoding on the training end!
            config["action_space_type"] = "discrete"
            config["state_space_size"] = 6
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

        assert config["sequence_length"] > 0, "config[\"sequence_length\"] <= 0. Set to: " + str(config["sequence_length"]) # also should be int
        if "completely_connected" in config and config["completely_connected"]:
            assert config["state_space_size"] == config["action_space_size"], "config[\"state_space_size\"] != config[\"action_space_size\"]. Please provide valid values! Vals: " + str(config["state_space_size"]) + " " + str(config["action_space_size"]) + ". In future, \"maximally_connected\" graphs are planned to be supported!"

        if "seed" in config:
            self.seed(config["seed"]) #seed
        else:
            self.seed()

        config["state_space_type"] = config["state_space_type"].lower()
        config["action_space_type"] = config["action_space_type"].lower()

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

        self.total_abs_noise_in_reward_episode = 0
        self.total_abs_noise_in_transition_episode = 0 # only present in continuous spaces
        self.total_reward_episode = 0
        self.total_noisy_transitions_episode = 0
        self.total_transitions_episode = 0

        # self.max_real = 100.0 # Take these settings from config
        # self.min_real = 0.0


        # def num_to_list(num1): #TODO Move this to a Utils file.
        #     if type(num1) == int or type(num1) == float:
        #         list1 = [num1]
        #     elif not isinstance(num1, list):
        #         raise TypeError("Argument to function should have been an int, float or list. Arg was: " + str(num1))
        #     return list1

        if config["state_space_type"] == "discrete":
            self.observation_space = DiscreteExtended(config["state_space_size"], seed=config["state_space_size"] + config["seed"]) #seed #hack #TODO Gym (and so Ray) apparently needs "observation"_space as a member. I'd prefer "state"_space
        else:
            self.state_space_max = config["state_space_max"]
            # config["state_space_max"] = num_to_list(config["state_space_max"]) * config["state_space_dim"]
            # print("config[\"state_space_max\"]", config["state_space_max"])

            self.observation_space = BoxExtended(-self.state_space_max, self.state_space_max, shape=(config["state_space_dim"], ), seed=10 + config["seed"], dtype=np.float64) #seed #hack #TODO

        if config["action_space_type"] == "discrete":
            self.action_space = DiscreteExtended(config["action_space_size"], seed=config["action_space_size"] + config["seed"]) #seed #hack #TODO
        else:
            self.action_space_max = config["action_space_max"]
            # config["action_space_max"] = num_to_list(config["action_space_max"]) * config["action_space_dim"]
            self.action_space = BoxExtended(-self.action_space_max, self.action_space_max, shape=(config["action_space_dim"], ), seed=11 + config["seed"], dtype=np.float64) #seed #hack #TODO

        if not config["generate_random_mdp"]:
            #TODO When having a fixed delay/specific sequences, we need to have transition function from tuples of states of diff. lengths to next tuple of states. We can have this tupleness to have Markovianness on 1 or both of dynamics and reward functions.
            # P(s, a) = s' for deterministic transitions; or a probability dist. function over s' for non-deterministic #TODO keep it a probability function for deterministic too? pdf would be Dirac-Delta at specific points! Can be done in sympy but not scipy I think. Or overload function to return Prob dist.(s') function when called as P(s, a); or pdf or pmf(s, a, s') value when called as P(s, a, s')
            self.P = config["transition_function"] if callable(config["transition_function"]) else lambda s, a: config["transition_function"][s, a] # callable may not be optimal always since it was deprecated in Python 3.0 and 3.1
            # R(s, a) or (s, a , s') = r for deterministic rewards; or a probability dist. function over r for non-deterministic and then the return value of P() is a function, too! #TODO What happens when config is out of scope? Maybe use as self.config?
            self.R = config["reward_function"] if callable(config["reward_function"]) else lambda s, a: config["reward_function"][s, a]
            ##### TODO self.P and R were untended to be used as the dynamics functions inside step() - that's why were being inited by user-defined function here; but currently P is being used as dynamics for imagined transitions and transition_function is used for actual transitions in step() instead. So, setting P to manually configured transition means it won't be used in step() as was intended!
        else:
            #TODO Generate state and action space sizes also randomly
            self.init_terminal_states()
            self.init_init_state_dist()
            self.init_reward_function()
            self.init_transition_function()

        #TODO make init_state_dist the default sample() for state space?
        if self.config["state_space_type"] == "discrete":
            self.init_state_dist = self.config["init_state_dist"] if callable(config["init_state_dist"]) else lambda s: config["init_state_dist"][s] #TODO make the probs. sum to 1 by using Sympy/mpmath? self.init_state_dist is not used anywhere right now, self.config["init_state_dist"] is used!
            print("self.init_state_dist:", self.config["init_state_dist"])
        #TODO sample at any time from "current"/converged distribution of states according to current policy
        self.curr_state = self.reset() #TODO make this seedable (Gym env has its own seed?); extend Discrete, etc. spaces to sample states at init or at any time acc. to curr. policy?;
        print("self.augmented_state, len", self.augmented_state, len(self.augmented_state))

        if self.config["state_space_type"] == "discrete":
            self.is_terminal_state = self.config["is_terminal_state"] if callable(self.config["is_terminal_state"]) else lambda s: s in self.config["is_terminal_state"]
            print("self.config['is_terminal_state']:", self.config["is_terminal_state"])
        else:
            self.is_terminal_state = lambda s: np.any([self.term_spaces[i].contains(s) for i in range(len(self.term_spaces))]) ### TODO for cont.


        if self.config["state_space_type"] == "discrete":
            delay = self.delay
            sequence_length = self.sequence_length
            self.possible_remaining_sequences = [[] for i in range(sequence_length)]
            for j in range(1):
            #        if j == 0:
                for k in range(sequence_length):
                    for l in range(len(self.specific_sequences[k])):
    #                    if state_considered[self.augmented_state_length - j - delay : self.augmented_state_length - delay] == self.specific_sequences[k][l][:j]: #TODO for variable sequence length just maintain a list of lists of lists rewarded_sequences
                            self.possible_remaining_sequences[j].append(self.specific_sequences[k][l][:j + 1])

            print("self.possible_remaining_sequences", self.possible_remaining_sequences)

            print(" self.delay, self.sequence_length:", self.delay, self.sequence_length)


        print("toy env instantiated with config:", self.config) #hack

        # Reward: Have some randomly picked sequences that lead to rewards (can make it sparse or non-sparse setting). Sequence length depends on how difficult we want to make it.
        # print("self.P:", np.array([[self.P(i, j) for j in range(5)] for i in range(5)]), self.config["transition_function"])
        # print("self.R:", np.array([[self.R(i, j) for j in range(5)] for i in range(5)]), self.config["reward_function"])
        # self.R = self.reward_function
        # print("self.R:", self.R(0, 1))

    def init_terminal_states(self):
        if self.config["state_space_type"] == "discrete":
            self.num_terminal_states = int(self.config["terminal_state_density"] * self.config["state_space_size"])
            if self.num_terminal_states == 0: # Have at least 1 terminal state
                print("WARNING: int(terminal_state_density * state_space_size) was 0. Setting num_terminal_states to be 1!")
                self.num_terminal_states = 1
            self.config["is_terminal_state"] = np.array([self.config["state_space_size"] - 1 - i for i in range(self.num_terminal_states)]) # terminal states inited to be at the "end" of the sorted states
            print("Inited terminal states to:", self.config["is_terminal_state"], "total", self.num_terminal_states)
        else: # if continuous space
            # print("#TODO for cont. spaces: term states")
            self.term_spaces = []
            for i in range(len(self.config["terminal_states"])):
                lows = np.array([self.config["terminal_states"][i][j] - self.config["term_state_edge"]/2 for j in range(self.config["state_space_dim"])])
                highs = np.array([self.config["terminal_states"][i][j] + self.config["term_state_edge"]/2 for j in range(self.config["state_space_dim"])])
                # print("Term state lows, highs:", lows, highs)
                self.term_spaces.append(BoxExtended(low=lows, high=highs, seed=12 + config["seed"], dtype=np.float64)) #seed #hack #TODO
            print("self.term_spaces samples:", self.term_spaces[0].sample(), self.term_spaces[-1].sample())

    def init_init_state_dist(self):
        if self.config["state_space_type"] == "discrete":
            self.config["init_state_dist"] = np.array([1 / (self.config["state_space_size"] - self.num_terminal_states) for i in range(self.config["state_space_size"] - self.num_terminal_states)] + [0 for i in range(self.num_terminal_states)]) #TODO Currently only uniform distribution over non-terminal states; Use Dirichlet distribution to select prob. distribution to use!
        else: # if continuous space
            # print("#TODO for cont. spaces: init_state_dist")
            pass # this is handled in reset where we resample if we sample a term. state

    def init_reward_function(self):
        #TODO Maybe refactor this code and put useful reusable permutation generators, etc. in one library
        #print(self.config["reward_function"], "init_reward_function")
        if self.config["state_space_type"] == "discrete":
            if self.config["repeats_in_sequences"]:
                num_possible_sequences = (self.observation_space.n - self.num_terminal_states) ** self.config["sequence_length"] #TODO if sequence cannot have replacement, use permutations; use state + action sequences? Subtracting the no. of terminal states from state_space size here to get "usable" states for sequences, having a terminal state even at end of reward sequence doesn't matter because to get reward we need to transition to next state which isn't possible for a terminal state.
                num_specific_sequences = int(self.config["reward_density"] * num_possible_sequences) #FIX Could be a memory problem if too large state space and too dense reward sequences
                self.specific_sequences = [[] for i in range(self.sequence_length)]
                sel_sequence_nums = self.np_random.choice(num_possible_sequences, size=num_specific_sequences, replace=False) #random # This assumes that all sequences have an equal likelihood of being selected for being a reward sequence;
                for i in range(num_specific_sequences):
                    curr_sequence_num = sel_sequence_nums[i]
                    specific_sequence = []
                    while len(specific_sequence) != self.config["sequence_length"]:
                        specific_sequence.append(curr_sequence_num % (self.observation_space.n - self.num_terminal_states))
                        curr_sequence_num = curr_sequence_num // (self.observation_space.n - self.num_terminal_states)
                    #bottleneck When we sample sequences here, it could get very slow if reward_density is high; alternative would be to assign numbers to sequences and then sample these numbers without replacement and take those sequences
                    # specific_sequence = self.observation_space.sample(size=self.config["sequence_length"], replace=True) # Be careful that sequence_length is less than state space size
                    self.specific_sequences[self.sequence_length - 1].append(specific_sequence) #hack
                    print("specific_sequence that will be rewarded", specific_sequence) #TODO impose a different distribution for these: independently sample state for each step of specific sequence; or conditionally dependent samples if we want something like DMPs/manifolds
                print("Total no. of rewarded sequences:", len(self.specific_sequences[self.sequence_length - 1]), "Out of", num_possible_sequences)
            else:
                state_space_size = self.config["state_space_size"] - self.num_terminal_states
                len_ = self.sequence_length
                permutations = list(range(state_space_size + 1 - len_, state_space_size + 1))
                print("Permutations order, 1 random number out of possible perms, no. of possible perms", permutations, np.random.randint(np.prod(permutations)), np.prod(permutations))
                num_possible_permutations = np.prod(permutations)
                num_specific_sequences = int(self.config["reward_density"] * num_possible_permutations)
                self.specific_sequences = [[] for i in range(self.sequence_length)]
                sel_sequence_nums = self.np_random.choice(num_possible_permutations, size=num_specific_sequences, replace=False) #random # This assumes that all sequences have an equal likelihood of being selected for being a reward sequence;
                total_false = 0
                for i in range(num_specific_sequences):
                    curr_permutation = sel_sequence_nums[i]
                    seq_ = []
                    curr_rem_digits = list(range(state_space_size))
                    for j in permutations[::-1]:
                        rem_ = curr_permutation % j
                        seq_.append(curr_rem_digits[rem_])
                        del curr_rem_digits[rem_]
                #         print("curr_rem_digits", curr_rem_digits)
                        curr_permutation = curr_permutation // j
                #         print(rem_, curr_permutation, j, seq_)
                #     print("T/F:", seq_ in self.specific_sequences)
                    if seq_ not in self.specific_sequences[self.sequence_length - 1]: #hack
                        total_false += 1 #TODO remove these extra checks
                    self.specific_sequences[self.sequence_length - 1].append(seq_)
                    print("specific_sequence that will be rewarded", seq_)
                #print(len(set(self.specific_sequences))) #error

                print("Number of generated sequences that did not clash with an existing one when it was generated:", total_false)
                print("Total no. of rewarded sequences:", len(self.specific_sequences[self.sequence_length - 1]), "Out of", num_possible_permutations)
        else: # if continuous space
            print("#TODO for cont. spaces: noise")

        self.R = lambda s, a: self.reward_function(s, a, only_query=False)


    def init_transition_function(self):

        # Future sequences don't depend on past sequences, only the next state depends on the past sequence of length, say n. n would depend on what order the dynamics are - 1st order would mean only 2 previous states needed to determine next state
        if self.config["state_space_type"] == "discrete":
            self.config["transition_function"] = np.zeros(shape=(self.config["state_space_size"], self.config["action_space_size"]), dtype=object)
            self.config["transition_function"][:] = -1 #IMP # To avoid having a valid value from the state space before we actually assign a usable value below!
            if self.config["completely_connected"]:
                for s in range(self.config["state_space_size"]):
                    self.config["transition_function"][s] = list(self.observation_space.sample(size=self.config["action_space_size"], replace=False)) #random #TODO Preferably use the seed of the Env for this?
            else:
                for s in range(self.config["state_space_size"]):
                    for a in range(self.config["action_space_size"]):
                        self.config["transition_function"][s, a] = self.observation_space.sample() #random #TODO Preferably use the seed of the Env for this?
            print(self.config["transition_function"], "init_transition_function", type(self.config["transition_function"][0, 0]))
        else: # if continuous space
            print("#TODO for cont. spaces")

        self.P = lambda s, a: self.transition_function(s, a, only_query = False)

    def reward_function(self, state, action, only_query=True): #TODO Make reward depend on state_action sequence instead of just state sequence? Maybe only use the action sequence for penalising action magnitude?
        delay = self.delay
        sequence_length = self.sequence_length
        reward = 0.0
        # print("TEST", self.augmented_state[0 : self.augmented_state_length - delay], state, action, self.specific_sequences, type(state), type(self.specific_sequences))
        state_considered = state if only_query else self.augmented_state # When we imagine a rollout, the user has to provide full augmented state as the argument!!
        if not isinstance(state_considered, list):
            state_considered = [state_considered] # to get around case when sequence is an int

        if self.config["state_space_type"] == "discrete":
            if not self.config["make_denser"]:
                if state_considered[0 : self.augmented_state_length - delay] in self.specific_sequences[self.sequence_length - 1]:
                    # print(state_considered, "with delay", self.config["delay"], "rewarded with:", 1)
                    reward += self.reward_unit
                else:
                    # print(state_considered, "with delay", self.config["delay"], "NOT rewarded.")
                    pass
            else:
                for j in range(1, sequence_length + 1):
            # Check if augmented_states - delay up to different lengths in the past are present in sequence lists of that particular length; if so add them to the list of length
                    curr_seq_being_checked = state_considered[self.augmented_state_length - j - delay : self.augmented_state_length - delay]
                    if curr_seq_being_checked in self.possible_remaining_sequences[j - 1]:
                        count_ = self.possible_remaining_sequences[j - 1].count(curr_seq_being_checked)
                        # print("curr_seq_being_checked, count in possible_remaining_sequences, reward", curr_seq_being_checked, count_, count_ * self.reward_unit * j / self.sequence_length)
                        reward += count_ * self.reward_unit * j / self.sequence_length #TODO Maybe make it possible to choose not to multiply by count_ as a config option

                self.possible_remaining_sequences = [[] for i in range(sequence_length)]
                for j in range(0, sequence_length):
            #        if j == 0:
                    for k in range(sequence_length):
                        for l in range(len(self.specific_sequences[k])):
                            if state_considered[self.augmented_state_length - j - delay : self.augmented_state_length - delay] == self.specific_sequences[k][l][:j]: #TODO for variable sequence length just maintain a list of lists of lists rewarded_sequences
                                self.possible_remaining_sequences[j].append(self.specific_sequences[k][l][:j + 1])

                print("rew", reward)
                print("self.possible_remaining_sequences", self.possible_remaining_sequences)

        else: # if continuous space
            # print("#TODO for cont. spaces: noise")
            if self.total_transitions_episode + 1 < self.augmented_state_length: # + 1 because even without transition there may be reward as R() is before P() in step()
                pass #TODO
            else:
                # print("######reward test", self.total_transitions_episode, np.array(self.augmented_state), np.array(self.augmented_state).shape)
                x = np.array(self.augmented_state)[0 : self.augmented_state_length - delay, 0]
                y = np.array(self.augmented_state)[0 : self.augmented_state_length - delay, 1]
                A = np.vstack([x, np.ones(len(x))]).T
                coeffs, sum_se, rank_A, singular_vals_A = np.linalg.lstsq(A, y, rcond=None)
                sum_se = sum_se[0]
                reward += (- np.sqrt(sum_se / self.sequence_length)) * self.reward_scale

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
        if self.config["state_space_type"] == "discrete":
            next_state = self.config["transition_function"][state, action]
            if "transition_noise" in self.config:
                probs = np.ones(shape=(self.config["state_space_size"],)) * self.config["transition_noise"] / (self.config["state_space_size"] - 1)
                probs[next_state] = 1 - self.config["transition_noise"]
                #TODO Samples according to new probs to get noisy discrete transition
                new_next_state = self.observation_space.sample(prob=probs) #random
                # print("noisy old next_state, new_next_state", next_state, new_next_state)
                if next_state != new_next_state:
                    print("NOISE inserted! old next_state, new_next_state", next_state, new_next_state)
                    self.total_noisy_transitions_episode += 1
                # print("new probs:", probs, self.observation_space.sample(prob=probs))
                next_state = new_next_state
                # assert np.sum(probs) == 1, str(np.sum(probs)) + " is not equal to " + str(1)

        else: # if continuous space
            # print("#TODO for cont. spaces: noise")
            if self.action_space.contains(action):
                ##TODO implement for multiple orders, currently only for 1st order systems.
                if self.dynamics_order == 1:
                    next_state = state + action * self.time_unit / self.inertia
            else:
                next_state = state
                print("WARNING: Action out of range of action space. Applying 0 action!!")
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

            # print("After transition", self.augmented_state)
        #TODO Check ergodicity of MDP/policy? Can calculate probability of a rewardable specific sequence occurring (for a random policy)
        # print("TEEEEEST:", self.config["transition_function"], [state, action])
        # if next_state in self.config["is_terminal_state"]:
        #     print("NEXT_STATE:", next_state, next_state in self.config["is_terminal_state"])
        return next_state

    def get_augmented_state():
        '''
        Intended to return the full state which would be Markovian. For noisy processes, this would need the noise distribution and random seed too?
        '''
        return {"curr_state": self.curr_state, "augmented_state": self.augmented_state}

    def reset(self):
        # TODO reset is also returning info dict to be able to return state in addition to observation;
        # TODO Do not start in a terminal state.
        if self.config["state_space_type"] == "discrete":
            self.curr_state = self.np_random.choice(self.config["state_space_size"], p=self.config["init_state_dist"]) #random
            print("RESET called. State reset to:", self.curr_state)
            self.augmented_state = [np.nan for i in range(self.augmented_state_length - 1)]
            self.augmented_state.append(self.curr_state)
            # self.augmented_state = np.array(self.augmented_state) # Do NOT make an np.array out of it because we want to test existence of the array in an array of arrays
        else: # if continuous space
            print("#TODO for cont. spaces: reset")
            while True: # Be careful about infinite loops
                term_space_was_sampled = False
                self.curr_state = self.observation_space.sample() #random
                for i in range(len(self.term_spaces)):
                    if self.term_spaces[i].contains(self.curr_state):
                        print("A state was sampled in term state subspace. Therefore, resampling. State was, subspace was:", self.curr_state, i) ##TODO Move this logic into a new class in Gym spaces that can contain subspaces for term states! (with warning/error if term subspaces cover whole state space)
                        term_space_was_sampled = True
                        break
                if not term_space_was_sampled:
                    break

            print("RESET called. State reset to:", self.curr_state)
            self.augmented_state = [[np.nan] * self.config["state_space_dim"] for i in range(self.augmented_state_length - 1)]
            self.augmented_state.append(self.curr_state)

        print("Noise stats for previous episode (total abs. noise in rewards, total abs. noise in transitions, total reward, total noisy transitions, total transitions):",
                self.total_abs_noise_in_reward_episode, self.total_abs_noise_in_transition_episode, self.total_reward_episode, self.total_noisy_transitions_episode,
                self.total_transitions_episode)
        self.total_abs_noise_in_reward_episode = 0
        self.total_abs_noise_in_transition_episode = 0 # only present in continuous spaces
        self.total_reward_episode = 0
        self.total_noisy_transitions_episode = 0
        self.total_transitions_episode = 0

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



# class OT4RL():
#     """Functions to perform calculations of metrics like Wasserstein distance. Useful for tracking learning progress by getting the distance between learnt and true models."""
#
#     def __init__(self, config = None):

def mean_sinkhorn_dist(model_1_P, model_2_P, model_1_R=None, model_2_R=None, weighted=True):
    """Calculates the mean approx. Wasserstein dist. between P(s, a) distributions over all the possible (s, a) values, each dist. weighted by the |R(s, a)| magnitudes. Can optionally be unweighted.
       If model_1_R=None and model_2_R=None, then model_1_P and model_2_P can be general models for which averaged approx. Wasserstein dist. (i.e. sinkhorn divergence) is calculated, for e.g., they could even be models of R().
    """


if __name__ == "__main__":

    config = {}
    config["seed"] = 0 #seed, 7 worked for initially sampling within term state subspace
    # config["state_space_type"] = "discrete"
    # config["action_space_type"] = "discrete"
    # config["state_space_size"] = 6
    # config["action_space_size"] = 6
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
    config["state_space_dim"] = 2
    config["action_space_dim"] = 2
    config["transition_dynamics_order"] = 1
    config["inertia"] = 1 # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
    config["state_space_max"] = 5 # Will be a Box in the range [-max, max]
    config["action_space_max"] = 1 # Will be a Box in the range [-max, max]
    config["time_unit"] = 1 # Discretization of time domain
    config["terminal_states"] = [[0.0, 1.0], [1.0, 0.0]]
    config["term_state_edge"] =  1.0 # Terminal states will be in a hypercube centred around the terminal states given above with the edge of the hypercube of this length.

    config["generate_random_mdp"] = True # This supersedes previous settings and generates a random transition function, a random reward function (for random specific sequences)
    config["delay"] = 1
    config["sequence_length"] = 10
    config["reward_scale"] = 1.0
#    config["transition_noise"] = 0.2 # Currently the fractional chance of transitioning to one of the remaining states when given the deterministic transition function - in future allow this to be given as function; keep in mind that the transition function itself could be made a stochastic function - does that qualify as noise though?
    # config["reward_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to reward function
    # config["transition_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to transition function in cont. spaces
    env = RLToyEnv(config)
#    env.seed(0)
#    from rl_toy.envs import RLToyEnv
#    env = gym.make("RLToy-v0")
#    print("env.spec.max_episode_steps, env.unwrapped:", env.spec.max_episode_steps, env.unwrapped)
    state = env.reset()
    # print("TEST", type(state))
    for _ in range(20):
        # env.render() # For GUI
        action = env.action_space.sample() # take a #random action # TODO currently DiscreteExtended returns a sampled array
        # action = np.array([1, 1]) # just to test if acting "in a line" works
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        state = next_state
    env.reset()
    env.close()

    # import sys
    # sys.exit(0)
