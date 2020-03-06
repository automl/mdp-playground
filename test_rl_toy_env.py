import unittest

# import os
# os.chdir(os.getcwd() + '/rl_toy/envs/')
# rl_toy_env = __import__('rl_toy_env')

import sys
sys.path.append('./rl_toy/envs/')
from rl_toy_env import RLToyEnv

import numpy as np
import copy

from datetime import datetime
log_filename = 'test_rl_toy_' + datetime.today().strftime('%m.%d.%Y_%I:%M:%S_%f') + '.log' #TODO Make a directoy 'log/' and store there.


#TODO None of the tests do anything when done = True. Should try calling reset() in one of them and see that this works?

class TestRLToyEnv(unittest.TestCase):

    def test_continuous_dynamics(self):
        '''
        '''
        print('\033[32;1;4mTEST_CONTINUOUS_DYNAMICS\033[0m')
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = {}
        config["seed"]["env"] = 0 # seed, 7 worked for initially sampling within term state subspace
        config["seed"]["state_space"] = 10
        config["seed"]["action_space"] = 11

        config["state_space_type"] = "continuous"
        config["action_space_type"] = "continuous"
        config["state_space_dim"] = 4
        config["action_space_dim"] = 4
        config["transition_dynamics_order"] = 1
        config["inertia"] = 1 # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
        config["time_unit"] = 1 # Discretization of time domain

        config["delay"] = 0
        config["sequence_length"] = 10
        config["reward_scale"] = 1.0
    #    config["transition_noise"] = 0.2 # Currently the fractional chance of transitioning to one of the remaining states when given the deterministic transition function - in future allow this to be given as function; keep in mind that the transition function itself could be made a stochastic function - does that qualify as noise though?
        # config["reward_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to reward function
        # config["transition_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to transition function in cont. spaces
        config["reward_function"] = "move_along_a_line"

        config["generate_random_mdp"] = True # This supersedes previous settings and generates a random transition function, a random reward function (for random specific sequences)

        # Test 1: general dynamics and reward
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        self.assertEqual(type(state), np.ndarray, "Type of continuous state should be numpy.ndarray.")
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([1, 1, 1, 1]) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(0.0, reward, atol=1e-7, err_msg='Step: ' + str(i))
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]))
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()
        env.close()


        # Test 2: sequence lengths #TODO


        # Test 3: that random actions lead to bad reward and then later a sequence of optimal actions leads to good reward. Also implicitly tests sequence lengths.
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for i in range(40):
            if i < 20:
                action = env.action_space.sample()
            else:
                action = np.array([1, 1, 1, 1])
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i >= 29:
                np.testing.assert_allclose(0.0, reward, atol=1e-7, err_msg='Step: ' + str(i))
            elif i >= 20: # reward should ideally start getting better at step 20 when we no longer apply random actions, but in this case, by chance, the 1st non-random action doesn't help
                assert prev_reward < reward, 'Step: ' + str(i) + ' Expected reward mismatch. Reward was: ' + str(reward) +  '. Prev. reward was: ' + str(prev_reward)
            elif i >= 9:
                assert reward < -1, 'Step: ' + str(i) + ' Expected reward mismatch. Reward was: ' + str(reward)
            state = next_state.copy()
            prev_reward = reward
        env.reset()
        env.close()


        # Test 4: same as 3 above except with delay
        print('\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_DELAY\033[0m')
        config["delay"] = 1
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for i in range(40):
            if i < 20:
                action = env.action_space.sample()
            else:
                action = np.array([1, 1, 1, 1])
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i >= 30:
                np.testing.assert_allclose(0.0, reward, atol=1e-7, err_msg='Step: ' + str(i))
            elif i >= 21:
                assert prev_reward < reward, 'Step: ' + str(i) + ' Expected reward mismatch. Reward was: ' + str(reward) +  '. Prev. reward was: ' + str(prev_reward)
            elif i >= 10:
                assert reward < -1, 'Step: ' + str(i) + ' Expected reward mismatch. Reward was: ' + str(reward)
            state = next_state.copy()
            prev_reward = reward
        env.reset()
        env.close()


        # Test 5: R noise - same as 1 above except with reward noise
        print('\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_R_NOISE\033[0m')
        config["reward_noise"] = lambda a: a.normal(0, 0.5)
        config["delay"] = 0
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        expected_rewards = [-0.70707351, 0.44681, 0.150735, -0.346204, 0.80687]
        for i in range(5):
            # action = env.action_space.sample()
            action = np.array([1, 1, 1, 1]) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(expected_rewards[i], reward, atol=1e-6, err_msg='Step: ' + str(i))
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([6.59339006, 5.68189965, 6.49608203, 5.19183292]), atol=1e-5)
        env.reset()
        env.close()


        # Test 6: for dynamics and reward in presence of irrelevant dimensions
        del config["reward_noise"]
        config["state_space_dim"] = 7
        config["action_space_dim"] = 7
        config["state_space_relevant_indices"] = [0, 1, 2, 6]
        config["action_space_relevant_indices"] = [0, 1, 2, 6]
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for i in range(20):
            action = env.action_space.sample()
            action[config["action_space_relevant_indices"]] = 1.0 # test to see if acting "in a line" works for relevant dimensions
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(0.0, reward, atol=1e-7, err_msg='Step: ' + str(i))
            state = next_state.copy()
        np.testing.assert_allclose(state[config["state_space_relevant_indices"]], np.array([21.59339006, 20.68189965, 21.49608203, 19.835966]))
        env.reset()
        env.close()

        # Test that random actions lead to bad reward in presence of irrelevant dimensions
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for i in range(20):
            action = env.action_space.sample()
            action[[3, 4, 5]] = 1.0 # test to see if acting "in a line" for relevant dimensions and not for relevant dimensions produces bad reward
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i > 10:
                assert reward < -0.8, 'Step: ' + str(i) + ' Expected reward mismatch. Reward was: ' + str(reward)
            state = next_state.copy()
        env.reset()
        env.close()


        # Test using config values: state_space_max and action_space_max
        config["state_space_max"] = 5 # Will be a Box in the range [-max, max]
        config["action_space_max"] = 1 # Will be a Box in the range [-max, max]
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for _ in range(20):
            # action = env.action_space.sample()
            action = np.array([-1] * 7) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([-5] * 7))
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()
        env.close()


        # Test for terminal states in presence of irrelevant dimensions
        config["terminal_states"] = [[0.92834036, 2.16924632, -4.88226269, -0.12869191], [2.96422742, -2.17263562, -2.71264267, 0.07446024]] # The 1st element is taken from the relevant dimensions of the default initial state for the given seed. This is to trigger a resample in reset. The 2nd element is taken from the relevant dimensions of the state reached after 2 iterations below. This is to trigger reaching a terminal state and a subsequent reset.
        config["term_state_edge"] = 1.0
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        state_derivatives = copy.deepcopy(env.state_derivatives)
        # augmented_state = copy.deepcopy(env.augmented_state)

        for _ in range(20):
            # action = env.action_space.sample()
            action = np.array([1] * 7) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(state_derivatives[0], env.augmented_state[-2]) # Tested here as well because
            state = next_state.copy()
            state_derivatives = copy.deepcopy(env.state_derivatives)
            # augmented_state = copy.deepcopy(env.augmented_state)
        np.testing.assert_allclose(state, np.array([5] * 7))
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()
        env.close()


        # Test P noise
        config["transition_noise"] = lambda a: a.normal([0] * 7, [0.5] * 7)
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        state_derivatives = copy.deepcopy(env.state_derivatives)
        # augmented_state = copy.deepcopy(env.augmented_state)

        expected_states = (np.array([ 1.96422742, -3.17263562, -3.71264267, -3.19641802,  0.09909165,
               -3.02478309, -0.92553976]),
                np.array([ 2.25715391, -1.72582608, -2.56190734, -2.5426217 ,  1.90596197,
                -2.53510777,  0.09614787]),
                np.array([ 2.90342939,  0.3748542 , -1.87656563, -1.48317271,  3.03932642,
               -1.08032816,  1.04361135]),)

        expected_noises = (np.array([-0.70707351,  0.44680953,  0.15073534, -0.34620368,  0.80687032,
                -0.51032468,  0.02168763]),
                np.array([-0.35372452,  1.10068028, -0.31465829,  0.05944899,  0.13336445,
                0.45477961, -0.05253652]),
                np.array([ 0.87593953, -0.32743438,  0.16137274,  0.20016199, -0.2355699 ,
                0.15253411, -0.85818094]),)

        for i in range(3):
            # action = env.action_space.sample()
            action = np.array([1] * 7) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(state_derivatives[0], env.augmented_state[-2]) # Tested here as well because
            state = next_state.copy()
            state_derivatives = copy.deepcopy(env.state_derivatives)
            np.testing.assert_allclose(state, expected_states[i] + expected_noises[i], err_msg='Failed at step: ' + str(i), rtol=1e-5)
            # augmented_state = copy.deepcopy(env.augmented_state)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()
        env.close()

### TODO Write test for continuous for checking reward with/without irrelevant dimensions, delay, r noise, seq_len?

    def test_continuous_dynamics_order(self):
        ''''''
        print('\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_ORDER\033[0m')
        config = {}
        config["log_filename"] = log_filename
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
        config["reward_function"] = "move_along_a_line"

        config["generate_random_mdp"] = True
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() # copy is needed to have a copy of the old state, otherwise we get the np.array that has the same location in memory and is constantly updated by step()
        state_derivatives = copy.deepcopy(env.state_derivatives)

        action = np.array([2.0, 1.0])
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        np.testing.assert_allclose(next_state - state, (1/6) * np.array([1, 0.5]) * 1e-6)
        np.testing.assert_allclose(env.state_derivatives[1] - state_derivatives[1], (1/2) * np.array([1, 0.5]) * 1e-4)
        np.testing.assert_allclose(env.state_derivatives[2] - state_derivatives[2], np.array([1, 0.5]) * 1e-2)
        np.testing.assert_allclose(state_derivatives[0], env.augmented_state[-2]) # Tested here as well because
        state = next_state.copy()
        state_derivatives = copy.deepcopy(env.state_derivatives)

        action = np.array([2.0, 1.0])
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        np.testing.assert_allclose(next_state - state, (7/6) * np.array([1, 0.5]) * 1e-6)
        np.testing.assert_allclose(env.state_derivatives[1] - state_derivatives[1], (3/2) * np.array([1, 0.5]) * 1e-4)
        np.testing.assert_allclose(env.state_derivatives[2] - state_derivatives[2], np.array([1, 0.5]) * 1e-2)
        np.testing.assert_allclose(state_derivatives[0], env.augmented_state[-2]) # Tested here as well because
        state = next_state.copy()

        #TODO Test for more timesteps? (>seq_len so that reward function kicks in) or higher order derivatives (.DONE)

        env.reset()
        env.close()


    def test_continuous_dynamics_target_point(self):
        ''''''
        print('\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_TARGET_POINT\033[0m')
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = {}
        config["seed"]["env"] = 3
        config["seed"]["state_space"] = 10000
        config["seed"]["action_space"] = 101

        config["state_space_type"] = "continuous"
        config["action_space_type"] = "continuous"
        config["state_space_dim"] = 2
        config["action_space_dim"] = 2
        config["transition_dynamics_order"] = 1
        config["inertia"] = 2.0
        config["time_unit"] = 0.1

        config["delay"] = 0
        config["sequence_length"] = 1 # seq_len is always going to be 1 for move_to_a_point R. assert for this?
        config["reward_scale"] = 1.0
        config["reward_function"] = "move_to_a_point"
        config["target_point"] = [-0.29792, 1.71012]
        config["make_denser"] = True

        config["generate_random_mdp"] = True

        # Test : dense reward
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([0.5]*2) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(0.035355, reward, atol=1e-6, err_msg='Step: ' + str(i)) # At each step, the distance reduces by ~0.035355 to the final point of this trajectory which is also the target point by design for this test. That is also the reward given at each step.
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([-0.29792, 1.71012]), atol=1e-6)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()
        env.close()

        # Test : sparse reward
        config["make_denser"] = False
        config["target_radius"] = 0.072 # to give reward in 3rd last step. At each step, the distance reduces by ~0.035355 to the final point of this trajectory which is also the target point by design for this test.
        config["reward_unit"] = 2.0
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state'].copy() #env.reset()
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([0.5]*2) # just to test if acting "in a line" works
            next_state, reward, done, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i < 17:
                np.testing.assert_allclose(0.0, reward, atol=1e-6, err_msg='Step: ' + str(i))
            else:
                np.testing.assert_allclose(2.0, reward, atol=1e-6, err_msg='Step: ' + str(i))
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([-0.29792, 1.71012]), atol=1e-6)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()
        env.close()

    def test_discrete_dynamics(self):
        ''''''
        config = {}
        config["log_filename"] = log_filename
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
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(next_state, 1, "Mismatch in state expected by transition dynamics for step 1.")
        state = next_state

        action = 4
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(next_state, 2, "Mismatch in state expected by transition dynamics for step 2.")
        state = next_state

        action = 1
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(next_state, 5, "Mismatch in state expected by transition dynamics for step 3.")
        self.assertEqual(done, True, "Mismatch in expectation that terminal state should have been reached by transition dynamics for step 3.")
        state = next_state

        # Try a random action to see that terminal state leads back to same terminal state
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(next_state, state, "Mismatch in state expected by transition dynamics for step 4. Terminal state was reached in step 3 and any random action should lead back to same terminal state.")
        state = next_state

        env.reset()
        env.close()


    def test_discrete_reward_delay(self):
        ''''''
        print('\033[32;1;4mTEST_DISCRETE_REWARD_DELAY\033[0m')
        config = {}
        config["log_filename"] = log_filename
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
        expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
        expected_states = [0, 2, 2, 5, 2, 5, 5, 0, 6]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            # self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.") # will not work for 2nd last time step due to random action.
            state = next_state

        env.reset()
        env.close()


    def test_discrete_rewardable_sequences(self):
        ''''''
        print('\033[32;1;4mTEST_DISCRETE_REWARDABLE_SEQUENCES\033[0m')
        config = {}
        config["log_filename"] = log_filename
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
        expected_rewards = [0, 1, 1, 0, 1, 0, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when sequence length = 3.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_p_noise(self):
        ''''''
        print('TEST_DISCRETE_P_NOISE')
        config = {}
        config["log_filename"] = log_filename
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
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(next_state, expected_states[i], "Expected next state mismatch in time step: " + str(i + 1) + " when P noise = 0.5.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_r_noise(self):
        ''''''
        print('TEST_DISCRETE_R_NOISE')
        config = {}
        config["log_filename"] = log_filename
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

        actions = [6, 6, 2, 1] #
        expected_rewards = [1 + -0.499716, 0.805124, -0.224812, 0.086749] # 2nd state produces 'true' reward
        for i in range(len(actions)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(reward, expected_rewards[i], rtol=1e-05, err_msg='Expected reward mismatch in time step: ' + str(i + 1) + ' when R noise = 0.5.')

            state = next_state

        env.reset()
        env.close()

###TODO Test for make_denser

    def test_discrete_all_meta_features(self):
        '''
        #TODO Currently only test for seq, del and r noises together. Include others! Gets complicated with P noise: trying to avoid terminal states while still following a rewardable sequence. Maybe try low P noise to test this?
        '''
        print('TEST_DISCRETE_ALL_META_FEATURES')

        config = {}
        config["log_filename"] = log_filename
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
        expected_rewards = [0 + -0.292808, 0 + 0.770696, 1 + -1.01743611, 1 + -0.042768, 0 + 0.78761320, 1 + -0.510087, 0 - 0.089978, 0 + 0.48654863]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
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
        config["log_filename"] = log_filename
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
        expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            state = next_state

        env.reset()
        env.close()


    def test_discrete_multi_discrete_irrelevant_dimensions(self):
        '''
        Same as the test_discrete_multi_discrete test above except with state_space_size and action_space_size having extra irrelevant dimensions
        '''
        print('\033[32;1;4mTEST_DISCRETE_MULTI_DISCRETE_IRRELEVANT_DIMENSIONS\033[0m')

        config = {}
        config["log_filename"] = log_filename
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
                print("sars', done =", state, actions[i], reward, next_state, done)
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
        expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
        expected_states = [[0, 0, 0, 3], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 3], [0, 1, 0, 2], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 0, 4], [1, 0, 0, 2]]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            state = next_state

        env.reset()
        env.close()


        # Test: This test lets even irrelevant dimensions be multi-dimensional
        config["state_space_size"] = [2, 2, 2, 1, 5]
        config["state_space_relevant_indices"] = [0, 1, 2]
        config["action_space_size"] = [2, 5, 1, 1, 2, 2]
        config["action_space_relevant_indices"] = [0, 4, 5]
        env = RLToyEnv(config)
        state = env.get_augmented_state()['curr_state']

        actions = [[1, 4, 0, 0, 1, 0], [0, 3, 0, 0, 1, 0], [1, 4, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 1], [0, 3, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 4, 0, 0, 0, 1], [1, 4, 0, 0, 0, 0]]
        expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
        expected_states = [[0, 0, 0, 0, 3], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 0, 3], [0, 1, 0, 0, 2], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 4], [1, 0, 0, 0, 2]]
        for i in range(len(expected_rewards)):
            next_state, reward, done, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
            state = next_state

        env.reset()
        env.close()


    #Unit tests


if __name__ == '__main__':
    unittest.main()
