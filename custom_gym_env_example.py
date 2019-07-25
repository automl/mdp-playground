
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box
from discrete_extended import DiscreteExtended


class CustomEnv(gym.Env):
    """Example of a custom env"""

    def __init__(self, config = None):
        """config can contain S, A, P, R, an initial distribution function over states (discrete or continuous), gamma?,
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
        #TODO Test cases to check all assumptions
        #TODO Mersenne Twister pseudo-random number generator used in Gym: not cryptographically secure!
        #Check TODO, fix and bottleneck
        """

        if config is None:
            config = {}
            config["state_space_type"] = "discrete" #TODO if states are assumed categorical in discrete setting, need to have an embedding for their OHE when using NNs.
            config["action_space_type"] = "discrete"
            config["state_space_size"] = 5
            config["action_space_size"] = 5

            # config["state_space_dim"] = 2
            # config["action_space_dim"] = 1
            config["transition_function"] = np.array([[4 - i for i in range(config["state_space_size"])] for j in range(config["state_space_size"])]) #TODO For all these prob. dist., there's currently a difference in what is returned for discrete vs continuous!
            config["reward_function"] = np.array([[4 - i for i in range(config["state_space_size"])] for j in range(config["state_space_size"])])
            config["init_state_dist"] = np.array([i/10 for i in range(config["state_space_size"])])
            config["is_terminal_state"] = np.array([config["state_space_size"] - 1]) # Can be discrete array or function to test terminal or not (e.g. for discrete and continuous spaces we may prefer 1 of the 2)

            config["generate_random_mdp"] = True
            config["delay"] = 1
            config["sequence_length"] = 2
            config["reward_density"] = 0.25 # Number between 0 and 1
            assert config["sequence_length"] > 0 # also should be int
            #next: To implement delay, we can keep the previous observations to make state Markovian or keep an info bit in the state to denote that; Buffer length increase by fixed delay and fixed sequence length; current reward is incremented when any of the satisfying conditions (based on previous states) matches

        self.config = config
        self.augmented_state_length = config["sequence_length"] + config["delay"]

        self.max_real = 100.0 # Take these settings from config
        self.min_real = 0.0

        if config["state_space_type"].lower() == "discrete":
            self.state_space = DiscreteExtended(config["state_space_size"])
        else:
            self.state_space = Box(self.min_real, self.max_real, shape=(config["state_space_dim"], ), dtype=np.float64)

        if config["action_space_type"].lower() == "discrete":
            self.action_space = DiscreteExtended(config["action_space_size"])
        else:
            self.action_space = Box(self.min_real, self.max_real, shape=(config["action_space_dim"], ), dtype=np.float64)

        if not config["generate_random_mdp"]:
            #TODO When having a fixed delay/specific sequences, we need to have transition function from tuples of states of diff. lengths to next tuple of states. We can have this tupleness to have Markovianness on 1 or both of dynamics and reward functions.
            # P(s, a) = s' for deterministic transitions; or a probability dist. function over s' for non-deterministic #TODO keep it a probability function for deterministic too? pdf would be Dirac-Delta at specific points! Can be done in sympy but not scipy I think. Or overload function to return Prob dist.(s') function when called as P(s, a); or pdf or pmf(s, a, s') value when called as P(s, a, s')
            self.P = config["transition_function"] if callable(config["transition_function"]) else lambda s, a: config["transition_function"][s, a] # callable may not be optimal always since it was deprecated in Python 3.0 and 3.1
            # R(s, a) or (s, a , s') = r for deterministic rewards; or a probability dist. function over r for non-deterministic and then the return value of P() is a function, too! #TODO What happens when config is out of scope? Maybe use as self.config?
            self.R = config["reward_function"] if callable(config["reward_function"]) else lambda s, a: config["reward_function"][s, a]
        else:
            self.init_reward_function()
            self.init_transition_function()


        # Reward: Have some randomly picked sequences that lead to rewards (can make it sparse or non-sparse setting). Sequence length depends on how difficult we want to make it.
        print("self.P:", np.array([[self.P(i, j) for j in range(5)] for i in range(5)]), self.config["transition_function"])
        print("self.R:", np.array([[self.R(i, j) for j in range(5)] for i in range(5)]), self.config["reward_function"])

        self.init_state_dist = config["init_state_dist"] # if callable(config["init_state_dist"]) else lambda s: config["init_state_dist"][s] #TODO make the probs. sum to 1 by using Sympy/mpmath?
        # print("self.init_state_dist:", self.init_state_dist)
        #TODO sample at any time from "current"/converged distribution of states according to current policy
        self.curr_state = np.random.choice(self.config["state_space_size"], p=self.init_state_dist) #TODO make this seedable (Gym env has its own seed?); extend Discrete, etc. spaces to sample states at init or at any time acc. to curr. policy?; Can't call reset() here because it has not been created yet!
        self.augmented_state = [np.nan for i in range(self.augmented_state_length - 1)]
        self.augmented_state.append(self.curr_state)
        self.augmented_state = np.array(self.augmented_state)
        print("self.augmented_state", self.augmented_state)
        self.is_terminal_state = self.config["is_terminal_state"] if callable(self.config["is_terminal_state"]) else lambda s: s in self.config["is_terminal_state"]
        print("self.config['is_terminal_state']:", self.config["is_terminal_state"])


    def init_reward_function(self):
        print(self.config["reward_function"], "init_reward_function")
        num_possible_sequences = self.action_space.n ** self.config["sequence_length"] #TODO if sequence cannot have replacement, use permutations
        num_specific_sequences = int(self.config["reward_density"] * num_possible_sequences) #FIX Could be a memory problem if too large state space and too dense reward sequences
        self.specific_sequences = []
        sel_sequence_nums = np.random.choice(num_possible_sequences, size=num_specific_sequences, replace=False) # This assumes that all sequences have an equal likelihood of being selected for being a reward sequence;
        for i in range(num_specific_sequences):
            curr_sequence_num = sel_sequence_nums[i]
            specific_sequence = []
            while len(specific_sequence) != self.config["sequence_length"]:
                specific_sequence.append(curr_sequence_num % self.state_space.n)
                curr_sequence_num = curr_sequence_num // self.state_space.n
            #bottleneck When we sample sequences here, it could get very slow if reward_density is high; alternative would be to assign numbers to sequences and then sample these numbers without replacement and take those sequences
            # specific_sequence = self.state_space.sample(size=self.config["sequence_length"], replace=True) # Be careful that sequence_length is less than state space size
            self.specific_sequences.append(specific_sequence)
            print("specific_sequence", specific_sequence, len(self.specific_sequences)) #TODO impose a different distribution for these: independently sample state for each step of specific sequence; or conditionally dependent samples if we want something like DMPs/manifolds


    def init_transition_function(self):
        print(self.config["transition_function"], "init_transition_function")


    def reward(self, state, action): #TODO Make reward depend on state_action sequence instead of just state sequence? Maybe only use the action sequence for penalising action magnitude?
        # if :
        pass

    def reset(self):
        #TODO reset is also returning info dict to be able to return state in addition to observation;
        self.curr_state = np.random.choice(self.config["state_space_size"], p=self.init_state_dist)
        return self.curr_state, {"curr_state": self.curr_state}

    def step(self, action):
        # assert self.action_space.contains(action) , str(action) + " not in" # TODO Need to implement check in for this environment
        #TODO check self.done and throw error if it's True? (Gym doesn't do it!) Otherwise, define self transitions in terminal states
        self.reward = self.R(self.curr_state, action)
        self.curr_state = self.P(self.curr_state, action)


        self.done = self.is_terminal_state(self.curr_state)
        return self.curr_state, self.reward, self.done, {"curr_state": self.curr_state}


if __name__ == "__main__":

    env = CustomEnv(None)
    state = env.reset()
    for _ in range(10):
        # env.render() # For GUI
        action = env.action_space.sample() # take a random action
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        state = next_state
    env.close()

    # import sys
    # sys.exit(0)
