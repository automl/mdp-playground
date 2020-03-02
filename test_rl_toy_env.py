import unittest

# import os
# os.chdir(os.getcwd() + '/rl_toy/envs/')
# rl_toy_env = __import__('rl_toy_env')

import sys
sys.path.append('./rl_toy/envs/')
from rl_toy_env import RLToyEnv

import numpy as np
import copy

class TestRLToyEnv(unittest.TestCase):

    def test_continuous_dynamics(self):
        ''''''
        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0 #seed, 7 worked for initially sampling within term state subspace
        config["seed"]["state_space"] = 10
        config["seed"]["action_space"] = 11

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
        self.assertEqual(type(state), np.ndarray, "Type of continuous state should be numpy.ndarray.")
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
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["state_space"] = 10
        config["seed"]["action_space"] = 11

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
        state_derivatives = copy.deepcopy(env.state_derivatives)

        action = np.array([2.0, 1.0])
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        np.testing.assert_allclose(next_state - state, (1/6) * np.array([1, 0.5]) * 1e-6)
        np.testing.assert_allclose(env.state_derivatives[1] - state_derivatives[1], (1/2) * np.array([1, 0.5]) * 1e-4)
        np.testing.assert_allclose(env.state_derivatives[2] - state_derivatives[2], np.array([1, 0.5]) * 1e-2)
        state = next_state.copy()
        state_derivatives = copy.deepcopy(env.state_derivatives)

        action = np.array([2.0, 1.0])
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        np.testing.assert_allclose(next_state - state, (7/6) * np.array([1, 0.5]) * 1e-6)
        np.testing.assert_allclose(env.state_derivatives[1] - state_derivatives[1], (3/2) * np.array([1, 0.5]) * 1e-4)
        np.testing.assert_allclose(env.state_derivatives[2] - state_derivatives[2], np.array([1, 0.5]) * 1e-2)
        state = next_state.copy()

        #TODO Test for more timesteps? or higher order derivatives (.DONE)

        env.reset()
        env.close()


    def test_discrete_dynamics(self):
        ''''''
        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 6
        config["seed"]["relevant_action_space"] = 6

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 6
        config["action_space_size"] = 6
        config["reward_density"] = 0.25
        config["make_denser"] = True
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 3
        config["reward_unit"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']
        self.assertEqual(type(state), int, "Type of discrete state should be int.") #TODO Move this and the test_continuous_dynamics type checks to separate unit tests

        action = 2
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        self.assertEqual(next_state, 1, "Mismatch in state expected by transition dynamics for step 1.")
        state = next_state

        action = 4
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        self.assertEqual(next_state, 2, "Mismatch in state expected by transition dynamics for step 2.")
        state = next_state

        action = 1
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        self.assertEqual(next_state, 5, "Mismatch in state expected by transition dynamics for step 3.")
        self.assertEqual(done, True, "Mismatch in expectation that terminal state should have been reached by transition dynamics for step 3.")
        state = next_state

        # Try a random action to see that terminal state leads back to same terminal state
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done, "\n")
        self.assertEqual(next_state, state, "Mismatch in state expected by transition dynamics for step 4. Terminal state was reached in step 3 and any random action should lead back to same terminal state.")
        state = next_state

        env.reset()
        env.close()


    def test_discrete_reward_delay(self):
        ''''''
        print('\033[31;1;4mTEST_DISCRETE_REWARD_DELAY\033[0m')
        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.25
        config["make_denser"] = True
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 3
        config["sequence_length"] = 1
        config["reward_unit"] = 1.0

        config["generate_random_mdp"] = True

        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [6, 2, 5, 4, 5, 2, 3, np.random.randint(config["action_space_size"]), 4] # 2nd last action is random just to check that last delayed reward works with any action
        expected_rewards = [0, 0, 0, 0, 1, 1, 0, 1, 0]
        expected_states = [0, 2, 2, 5, 2, 5, 5, 0, 6]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            # self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.") # will not work for 2nd last time step due to random action.
            state = next_state

        env.reset()
        env.close()


    def test_discrete_rewardable_sequences(self):
        ''''''
        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.25
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 3
        config["reward_unit"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [6, 6, 2, 3, 4, 2, np.random.randint(config["action_space_size"]), 5] #
        expected_rewards = [0, 0, 1, 1, 0, 1, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when sequence length = 3.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_p_noise(self):
        ''''''
        print('TEST_DISCRETE_P_NOISE')
        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.25
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 1
        config["reward_unit"] = 1.0
        config["transition_noise"] = 0.5

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [6, 6, 2, np.random.randint(config["action_space_size"])] #
        expected_states = [2, 6, 6, 3] # Last state 3 is fixed for this test because of fixed seed for Env which selects the next noisy state.
        for i in range(len(actions)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            self.assertEqual(next_state, expected_states[i], "Expected next state mismatch in time step: " + str(i + 1) + " when P noise = 0.5.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_r_noise(self):
        ''''''
        print('TEST_DISCRETE_R_NOISE')
        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.25
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 1
        config["reward_unit"] = 1.0
        config["reward_noise"] = lambda a: a.normal(0, 0.5)

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [6, 6, 2, np.random.randint(config["action_space_size"])] #
        expected_rewards = [-0.499716, 1.805124, -0.224812, 0.086749] # 2nd state produces 'true' reward
        for i in range(len(actions)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            np.testing.assert_allclose(reward, expected_rewards[i], rtol=1e-05, err_msg='Expected reward mismatch in time step: ' + str(i + 1) + ' when R noise = 0.5.')

            state = next_state

        env.reset()
        env.close()


    def test_discrete_all_meta_features(self):
        '''
        #TODO Currently only test for seq, del and r noises together. Include others! Gets complicated with P noise: trying to avoid terminal states while still following a rewardable sequence. Maybe try low P noise to test this?
        '''
        print('TEST_DISCRETE_ALL_META_FEATURES')

        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.25
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 1
        config["sequence_length"] = 3
        config["reward_unit"] = 1.0
        # config["transition_noise"] = 0.1
        config["reward_noise"] = lambda a: a.normal(0, 0.5)

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [6, 6, 2, 3, 4, 2, np.random.randint(config["action_space_size"]), 5] #
        expected_rewards = [0 + -0.292808, 0 + 0.770696, 0 + -1.01743611, 1 + -0.042768, 1 + 0.78761310, 0 + -0.510087, 1 - 0.089978, 1 - 0.51345136]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            np.testing.assert_allclose(reward, expected_rewards[i], rtol=1e-05, err_msg="Expected reward mismatch in time step: " + str(i + 1) + " when sequence length = 3, delay = 1.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_multi_discrete(self):
        '''
        Same as the test_discrete_reward_delay test above except with state_space_size and action_space_size specified as vectors and the actions slightly different near the end.
        '''
        print('TEST_DISCRETE_MULTI_DISCRETE')

        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = [2, 2, 2]
        config["state_space_relevant_indices"] = [0, 1, 2]
        config["action_space_size"] = [2, 2, 2]
        config["action_space_relevant_indices"] = [0, 1, 2]
        config["reward_density"] = 0.25
        config["make_denser"] = True
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 3
        config["sequence_length"] = 1
        config["reward_unit"] = 1.0

        config["generate_random_mdp"] = True

        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [[1, 1, 0], [0, 1, 0], [1, 0 ,1], [1, 0 ,0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]]
        expected_rewards = [0, 0, 0, 0, 1, 1, 0, 1, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_multi_discrete_irrelevant_dimensions(self):
        '''
        Same as the test_discrete_multi_discrete test above except with state_space_size and action_space_size having extra irrelevant dimensions
        '''
        print('\033[31;1;4mTEST_DISCRETE_MULTI_DISCRETE_IRRELEVANT_DIMENSIONS\033[0m')

        config = {}
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8
        config["seed"]["irrelevant_state_space"] = 52
        config["seed"]["irrelevant_action_space"] = 65
        config["seed"]["state_space"] = 87
        config["seed"]["action_space"] = 89

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = [2, 2, 2, 3]
        config["state_space_relevant_indices"] = [0, 1, 2]
        config["action_space_size"] = [2, 5, 2, 2]
        config["action_space_relevant_indices"] = [0, 2, 3]
        config["reward_density"] = 0.25
        config["make_denser"] = True
        config["terminal_state_density"] = 0.25
        config["completely_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 3
        config["sequence_length"] = 1
        config["reward_unit"] = 1.0

        config["generate_random_mdp"] = True

        try: # Testing for completely_connected options working properly when invalid config specified. #TODO Is this part needed?
            env = RLToyEnv(config)
            state = env.get_augmented_state()['curr_state']

            actions = [[1, 1, 0], [0, 1, 0], [1, 0 ,1], [1, 0 ,0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]]
            expected_rewards = [0, 0, 0, 0, 1, 1, 0, 1, 0]
            for i in range(len(expected_rewards)):
                next_state, reward, done, info = env.step(actions[i])
                print("sars', done =", state, actions[i], reward, next_state, done, "\n")
                self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
                state = next_state

            env.reset()
            env.close()

        except AssertionError as e:
            print('Caught Expected exception:', e)


        # Test: Adds one irrelevant dimension
        config["state_space_size"] = [2, 2, 2, 5]
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [[1, 4, 1, 0], [0, 3, 1, 0], [1, 4, 0, 1], [1, 0 ,0, 0], [1, 2, 0, 1], [0, 3, 1, 0], [0, 1, 1, 1], [0, 4, 0, 1], [1, 4, 0, 0]]
        expected_rewards = [0, 0, 0, 0, 1, 1, 0, 1, 0]
        expected_states = [[0, 0, 0, 3], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 3], [0, 1, 0, 2], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 0, 4], [1, 0, 0, 2]]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            state = next_state

        env.reset()
        env.close()


        # Test: Lets even irrelevant dimensions be multi-dimensional
        config["state_space_size"] = [2, 2, 2, 1, 5]
        config["state_space_relevant_indices"] = [0, 1, 2]
        config["action_space_size"] = [2, 5, 1, 1, 2, 2]
        config["action_space_relevant_indices"] = [0, 4, 5]
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [[1, 4, 0, 0, 1, 0], [0, 3, 0, 0, 1, 0], [1, 4, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 1], [0, 3, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 4, 0, 0, 0, 1], [1, 4, 0, 0, 0, 0]]
        expected_rewards = [0, 0, 0, 0, 1, 1, 0, 1, 0]
        expected_states = [[0, 0, 0, 0, 3], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 0, 3], [0, 1, 0, 0, 2], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 4], [1, 0, 0, 0, 2]]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done, "\n")
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            state = next_state

        env.reset()
        env.close()


    #Unit tests


if __name__ == '__main__':
    unittest.main()
