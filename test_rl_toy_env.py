import unittest

# import os
# os.chdir(os.getcwd() + '/rl_toy/envs/')
# rl_toy_env = __import__('rl_toy_env')

import sys
sys.path.append('./rl_toy/envs/')
from rl_toy_env import RLToyEnv

import numpy as np

class TestRLToyEnv(unittest.TestCase):

    def test_continuous_dynamics(self):
        ''''''
        config = {}
        config["seed"] = 0 #seed, 7 worked for initially sampling within term state subspace

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
        state = env.get_augmented_state()['curr_state'] #env.reset()
        self.assertEqual(type(state), np.ndarray, "Type of state should be numpy.ndarray.")
        for _ in range(20):
            # action = env.action_space.sample()
            action = np.array([1, 1, 1, 1]) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done, "\n")
            state = next_state
        np.testing.assert_allclose(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183192]))
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183192]), places=3) # Error
        env.reset()
        env.close()

    def test_continuous_dynamics_order(self):
        ''''''
        config = {}
        config["seed"] = 0

        config["state_space_type"] = "continuous"
        config["action_space_type"] = "continuous"
        config["state_space_dim"] = 2
        config["action_space_dim"] = 2
        config["transition_dynamics_order"] = 3
        config["inertia"] = 2.0
        config["time_unit"] = 0.01

        config["delay"] = 0
        config["sequence_length"] = 3
        config["reward_scale"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() # copy is needed to have a copy of the old state, otherwise we get the np.array that has the same location in memory and is constantly updated by step()

        action = np.array([2.0, 1.0])
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        np.testing.assert_allclose(next_state - state, (1/6) * np.array([1, 0.5]) * 1e-6)
        state = next_state.copy()

        action = np.array([2.0, 1.0])
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        np.testing.assert_allclose(next_state - state, (7/6) * np.array([1, 0.5]) * 1e-6)
        state = next_state.copy()

        env.reset()
        env.close()


    def test_discrete_dynamics(self):
        ''''''
        config = {}
        config["seed"] = 0

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 6
        config["action_space_size"] = 6
        config["reward_density"] = 0.25
        config["make_denser"] = True
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 1
        config["sequence_length"] = 3
        config["reward_unit"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']
        self.assertEqual(type(state), int, "Type of state should be numpy.ndarray.") #TODO Move this and the test_continuous_dynamics type checks to separate unit tests

        action = 1
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        self.assertEqual(next_state, 5, "Mismatch in state expected by transition dynamics for step 1.")
        state = next_state

        action = 5
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        self.assertEqual(next_state, 1, "Mismatch in state expected by transition dynamics for step 2.")
        state = next_state

        #TODO Test for more timesteps or higher order derivatives

        env.reset()
        env.close()


    #Unit tests


if __name__ == '__main__':
    unittest.main()
