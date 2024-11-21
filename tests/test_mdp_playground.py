import sys
from datetime import datetime
import logging
import copy
import numpy as np
from mdp_playground.envs.rl_toy_env import RLToyEnv
import unittest

# import os
# os.chdir(os.getcwd() + '/mdp_playground/envs/')
# rl_toy_env = __import__('rl_toy_env')

# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, "./mdp_playground/envs/")
# sys.path.append('./mdp_playground/envs/')


log_filename = (
    "/tmp/test_mdp_playground_"
    + datetime.today().strftime("%m.%d.%Y_%I:%M:%S_%f")
    + ".log"
)  # TODO Make a directoy 'log/' and store there.


np.set_printoptions(formatter={"float_kind": lambda x: "%.6g" % x})

# TODO None of the tests do anything when done = True. Should try calling reset() in one of them and see that this works?


class TestRLToyEnv(unittest.TestCase):
    def test_continuous_dynamics_move_along_a_line(self):
        """ """
        print("\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_MOVE_ALONG_A_LINE\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = {}
        config["seed"][
            "env"
        ] = 0  # seed, 7 worked for initially sampling within term state subspace
        config["seed"]["state_space"] = 10
        config["seed"]["action_space"] = 11

        config["state_space_type"] = "continuous"
        config["action_space_type"] = "continuous"
        config["state_space_dim"] = 4
        config["action_space_dim"] = 4
        config["transition_dynamics_order"] = 1
        config[
            "inertia"
        ] = 1  # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
        config["time_unit"] = 1  # Discretization of time domain

        config["delay"] = 0
        config["sequence_length"] = 10
        config["reward_scale"] = 1.0
        #    config["transition_noise"] = 0.2 # Currently the fractional chance of transitioning to one of the remaining states when given the deterministic transition function - in future allow this to be given as function; keep in mind that the transition function itself could be made a stochastic function - does that qualify as noise though?
        # config["reward_noise"] = lambda a: a.normal(0, 0.1) #random #hack # a probability function added to reward function
        # config["transition_noise"] = lambda a: a.normal(0, 0.1) #random #hack #
        # a probability function added to transition function in cont. spaces
        config["reward_function"] = "move_along_a_line"

        # Test 1: general dynamics and reward
        print("\nTest 1: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_GENERAL\033[0m")
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        self.assertEqual(
            type(state), np.ndarray, "Type of continuous state should be numpy.ndarray."
        )
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([1, 1, 1, 1], dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            # Since the actions move the agent along a line, expected reward is 0:
            np.testing.assert_allclose(
                0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
            )
            state = next_state.copy()
        np.testing.assert_allclose(
            state, np.array([18.896662, 19.274975, 19.218195, 20.266975])
        )
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

        # Test 2: sequence lengths # TODO done in "Test 3".

        # Test 3: that random actions lead to bad reward and then later a sequence
        # of optimal actions leads to good reward. Also implicitly tests sequence
        # lengths.
        print("\nTest 3: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_RANDOM_ACTIONS\033[0m")
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        prev_reward = None
        for i in range(40):
            if i < 20:
                action = env.action_space.sample()
            else:
                action = np.array([1, 1, 1, 1], dtype=env.action_space.dtype)
            next_state, reward, done, trunc, info = env.step(action)
            print(i, "sars', done =", state, action, reward, next_state, done)
            if i >= 29:
                np.testing.assert_allclose(
                    0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            elif (
                i >= 20
            ):  # reward should ideally start getting better at step 20 when we no longer apply random actions,
            # but in the previous case, by chance, the 1st non-random action didn't increase the reward
            # and in the current test case this occured at step 24, so I have added a buffer of 0.05 when comparing
            # the previous and current rewards. Not sure about the best way to handle this, in general.
                assert prev_reward < reward + 0.05, (
                    "Step: "
                    + str(i)
                    + " Expected reward mismatch. Reward was: "
                    + str(reward)
                    + ". Prev. reward was: "
                    + str(prev_reward)
                )
            elif i >= 9:
                # Had printed out the reward values along the random action trajectory
                # and observed that -0.92 was the max among these values, so using -0.9.
                assert reward < -0.9, (
                    "Step: "
                    + str(i)
                    + " Expected reward mismatch. Reward was: "
                    + str(reward)
                )
            state = next_state.copy()
            prev_reward = reward
        env.reset()[0]
        env.close()

        # Test 4: same as 3 above except with delay
        print("\nTest 4: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_DELAY\033[0m")
        config["delay"] = 1
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        prev_reward = None
        for i in range(40):
            if i < 20:
                action = env.action_space.sample()
            else:
                action = np.array([1, 1, 1, 1], dtype=env.action_space.dtype)
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i >= 30:
                np.testing.assert_allclose(
                    0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            elif i >= 21:
                assert prev_reward < reward + 0.05, (
                    "Step: "
                    + str(i)
                    + " Expected reward mismatch. Reward was: "
                    + str(reward)
                    + ". Prev. reward was: "
                    + str(prev_reward)
                )
            elif i >= 10:
                assert reward < -0.9, (
                    "Step: "
                    + str(i)
                    + " Expected reward mismatch. Reward was: "
                    + str(reward)
                )
            state = next_state.copy()
            prev_reward = reward
        env.reset()[0]
        env.close()

        # Test 5: R noise - same as Test 1 above except with reward noise and with only 5 steps
        # instead of 20.
        print("\nTest 5: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_R_NOISE\033[0m")
        config["reward_noise"] = lambda s, a, rng: rng.normal(0, 0.5)
        config["delay"] = 0
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        expected_rewards = [0.062865110, -0.0660524, 0.32021132, 0.05245005, -0.267834]
        for i in range(5):
            # action = env.action_space.sample()
            action = np.array([1, 1, 1, 1], dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(
                expected_rewards[i], reward, atol=1e-5, err_msg="Step: " + str(i)
            )
            state = next_state.copy()
        np.testing.assert_allclose(
            state, np.array([3.896662, 4.274975, 4.218195, 5.266975]), atol=1e-5
        )
        env.reset()[0]
        env.close()

        # Test 6: for dynamics and reward in presence of irrelevant dimensions
        print("\nTest 6: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_IRR_DIMS\033[0m")
        del config["reward_noise"]
        config["state_space_dim"] = 7
        config["action_space_dim"] = 7
        config["relevant_indices"] = [0, 1, 2, 6]
        config["action_space_relevant_indices"] = [0, 1, 2, 6]
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            action = env.action_space.sample()
            action[
                config["action_space_relevant_indices"]
            ] = 1.0  # test to see if acting "in a line" works for relevant dimensions
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(
                0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
            )
            state = next_state.copy()
        np.testing.assert_allclose(
            state[config["relevant_indices"]],
            np.array([18.8967, 19.275, 19.2182, 20.843]), atol=1e-4
        )
        env.reset()[0]
        env.close()

        # Test that random actions in relevant action space along with linear
        # actions in irrelevant action space leads to bad reward for
        # move_along_a_line reward function
        print("\nTest 7: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_IRR_DIMS_2\033[0m")
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            action = env.action_space.sample()
            # test to see if acting "in a line" for irrelevant dimensions and not for relevant dimensions produces bad reward
            action[[3, 4, 5]] = 1.0
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i > 10:
                assert reward < -0.8, (
                    "Step: "
                    + str(i)
                    + " Expected reward mismatch. Reward was: "
                    + str(reward)
                )
            state = next_state.copy()
        env.reset()[0]
        env.close()

        # Test using config values: state_space_max and action_space_max
        print("\nTest 8: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_SPACE_MAXES\033[0m")
        config["state_space_max"] = 5  # Will be a Box in the range [-max, max]
        config["action_space_max"] = 1  # Will be a Box in the range [-max, max]
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([-1] * 7, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([-5] * 7))
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

        # Test for terminal states in presence of irrelevant dimensions. In the foll.
        # array, the 1st element is taken from the relevant dimensions of the default
        # initial state for the given seed. This is to trigger a resample in
        # reset. The 2nd element is taken from the relevant dimensions of the
        # state reached after 2 iterations below. This is to trigger reaching a
        # terminal state. The trajectory is continued even after a terminal state
        # is reached and MDP Playground allows this. States reached afterwards are not
        # terminal.
        print("\nTest 9: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_IRR_DIMS_TERM_STATES\033[0m")
        config["terminal_states"] = [
            [4.56002, -2.92318, 3.28445, 1.89036],
            [5, 1.25509, 5, 4.53302],
        ]
        config["term_state_edge"] = 1.0
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        state_derivatives = copy.deepcopy(env.state_derivatives)
        # augmented_state = copy.deepcopy(env.augmented_state)

        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([1] * 7, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i == 1:
                assert done, "Terminal state should have been reached at step " + str(i) + "but was not."
            # augmented_state has "seq_len" num. of last states in a moving window, so the -1th state is the next_state
            # and -2th state is the current state.
            np.testing.assert_allclose(
                state_derivatives[0], env.augmented_state[-2]
            )  # Tested here as well because
            state = next_state.copy()
            state_derivatives = copy.deepcopy(env.state_derivatives)
            # augmented_state = copy.deepcopy(env.augmented_state)
        np.testing.assert_allclose(
            state, np.array([5] * 7)
        )  # It will be 5 because of being clipped once it is > state_space_max
        env.reset()[0]
        env.close()

        # Test P noise
        print("\nTest 9: \033[32;1;4mTEST_CONTINUOUS_DYNAMICS_P_NOISE\033[0m")
        config["transition_noise"] = lambda s, a, rng: rng.normal([0] * 7, [0.5] * 7)
        # Reset seed to have states far away from state maxes so that it is easier to 
        # test stuff below, but in the end, the state is clipped to [-5, 5] anyway
        # while testing, so this wasn't really needed.
        config["seed"] = 12
        env = RLToyEnv(**config)
        # env.reset()
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        state_derivatives = copy.deepcopy(env.state_derivatives)
        # augmented_state = copy.deepcopy(env.augmented_state)

        # All the following are +1ed because the next states are the current noiseless
        # states + 1
        expected_states = (
            np.array(
                [
                    1.74915,
                    -4.23287,
                    -1.28455,
                    2.31401,
                    2.13407,
                    -3.91503,
                    -0.662919
                ]
            ) + 1.0,
            np.array(
                [
                    2.08882,
                    -3.28675,
                    0.214836,
                    3.30304,
                    3.38201,
                    -3.87041,
                    0.410613
                ]
            ) + 1.0,
            np.array(
                [
                    2.63535,
                    -1.39905,
                    1.65826,
                    4.77771,
                    4.35309,
                    -2.56398,
                    1.73956
                ]
            ) + 1.0,
        )

        expected_noises = (
            np.array(  
                [
                    -0.660332,
                    -0.0538763,
                    0.499382,
                    -0.0109739,
                    0.24794,
                    -0.955384,
                    0.0735321
                ]            
                ),
            np.array(
                [
                    -0.453472,
                    0.887695,
                    0.443425,
                    0.474675,
                    -0.0289275,
                    0.306431,
                    0.328945
                ]
            ),
            np.array(
                [
                    -0.172201,
                    -0.248686,
                    -0.0573864,
                    -0.302726,
                    -0.29717,
                    -0.141688,
                    -0.364209
                ]
            ),
        )

        for i in range(3):
            # action = env.action_space.sample()
            action = np.array([1] * 7, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            np.testing.assert_allclose(
                state_derivatives[0], env.augmented_state[-2]
            )  # Tested here as well because
            state = next_state.copy()
            state_derivatives = copy.deepcopy(env.state_derivatives)
            np.testing.assert_allclose(
                state,
                np.clip(expected_states[i] + expected_noises[i], -5, 5),
                err_msg="Failed at step: " + str(i),
                rtol=1e-4,
            )
            # augmented_state = copy.deepcopy(env.augmented_state)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

    # TODO Write test for continuous for checking reward with/without irrelevant dimensions, delay, r noise, seq_len?

    def test_continuous_dynamics_order(self):
        """"""
        print("\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_ORDER\033[0m")
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

        env = RLToyEnv(**config)
        # copy is needed to have a copy of the old state, otherwise we get the
        # np.array that has the same location in memory and is constantly updated
        # by step()
        state = env.get_augmented_state()["curr_state"].copy()
        state_derivatives = copy.deepcopy(env.state_derivatives)

        action = np.array([2.0, 1.0], dtype=env.action_space.dtype)
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        # The following comparisons are based on the equations of motion for order 3,
        # time unit 0.01 and inertia 2:
        np.testing.assert_allclose(
            next_state - state, (1 / 6) * np.array([1, 0.5]) * 1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            env.state_derivatives[1] - state_derivatives[1],
            (1 / 2) * np.array([1, 0.5]) * 1e-4,
        )
        np.testing.assert_allclose(
            env.state_derivatives[2] - state_derivatives[2], np.array([1, 0.5]) * 1e-2
        )
        np.testing.assert_allclose(
            state_derivatives[0], env.augmented_state[-2]
        )  # Tested here as well because
        state = next_state.copy()
        state_derivatives = copy.deepcopy(env.state_derivatives)

        action = np.array([2.0, 1.0], dtype=env.action_space.dtype)
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        np.testing.assert_allclose(
            next_state - state, (7 / 6) * np.array([1, 0.5]) * 1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            env.state_derivatives[1] - state_derivatives[1],
            (3 / 2) * np.array([1, 0.5]) * 1e-4,
        )
        np.testing.assert_allclose(
            env.state_derivatives[2] - state_derivatives[2], np.array([1, 0.5]) * 1e-2
        )
        np.testing.assert_allclose(
            state_derivatives[0], env.augmented_state[-2]
        )  # Tested here as well because
        state = next_state.copy()

        # TODO Test for more timesteps? (>seq_len so that reward function kicks in) or higher order derivatives (.DONE)

        env.reset()[0]
        env.close()

    def test_continuous_dynamics_target_point_dense(self):
        """"""
        print("\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_TARGET_POINT_DENSE\033[0m")
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
        config[
            "sequence_length"
        ] = 1  # seq_len is always going to be 1 for move_to_a_point R. assert for this? #TODO
        config["reward_scale"] = 1.0
        config["reward_function"] = "move_to_a_point"
        config["target_point"] = [0.69422, 1.27494]
        config["target_radius"] = 0.05
        config["make_denser"] = True

        # Test : dense reward
        # The object moves with inertia 2 and time_unit 0.1 towards the chosen target in equal steps
        # with the same action at every step, so the reward is pretty much constant (except for numerical issues
        # or at the last step I think)
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([0.5] * 2, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print(i, "sars', done =", state, action, reward, next_state, done)
            # At each step, the distance reduces by ~0.035355 to the final point of
            # this trajectory which is also the target point by design for this test.
            # That is also the reward given at each step.
            np.testing.assert_allclose(
                0.0353553, reward, atol=1e-5, err_msg="Step: " + str(i)
            )
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([0.69422, 1.27494], dtype=env.observation_space.dtype), atol=1e-5)
        env.reset()[0]
        env.close()

        # Test irrelevant dimensions
        # The relevant indices are 1 and 2 not 0 and 1, the rest is the same as for the above test.
        config["state_space_dim"] = 5
        config["action_space_dim"] = 5
        config["relevant_indices"] = [1, 2]
        config["action_space_relevant_indices"] = [1, 2]
        config["target_point"] = [1.27494, -0.780999]
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([0.5] * 5, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print(i, "sars', done =", state, action, reward, next_state, done)
            # At each step, the distance reduces by ~0.035355 to the final point of
            # this trajectory which is also the target point by design for this test.
            # That is also the reward given at each step.
            np.testing.assert_allclose(
                0.035355, reward, atol=1e-5, err_msg="Step: " + str(i)
            )
            state = next_state.copy()
        np.testing.assert_allclose(
            state,
            np.array([0.69422, 1.27494, -0.780999, 1.52398, -0.311794]),
            atol=1e-5,
        )
        # check 1 extra step away from target point gives -ve reward
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        # At each step, the distance reduces by ~0.035355 to the final point of
        # this trajectory which is also the target point by design for this
        np.testing.assert_allclose(
            -0.035355, reward, atol=1e-5, err_msg="Step: " + str(i)
        )
        env.reset()[0]
        env.close()

        # Test delay
        config["delay"] = 10
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([0.5] * 5, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i < 10:
                np.testing.assert_allclose(
                    0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )  # delay part
            else:
                # At each step, the distance reduces by ~0.035355 to the final point of
                # this trajectory which is also the target point by design for this test.
                # That is also the reward given at each step.
                np.testing.assert_allclose(
                    0.035355, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            state = next_state.copy()
        np.testing.assert_allclose(
            state,
            np.array([0.69422, 1.27494, -0.780999, 1.52398, -0.311794]),
            atol=1e-5,
        )
        env.reset()[0]
        env.close()

    def test_continuous_dynamics_target_point_sparse(self):
        """"""
        print("\033[32;1;4mTEST_CONTINUOUS_DYNAMICS_TARGET_POINT_SPARSE\033[0m")
        config = {}
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
        config[
            "sequence_length"
        ] = 1  # seq_len is always going to be 1 for move_to_a_point R. assert for this?
        config["reward_function"] = "move_to_a_point"
        config["make_denser"] = False
        config["target_point"] = [0.69422, 1.27494]
        # to give reward in 3rd last step. At each step, the distance reduces by
        # ~0.035355 to the final point of this trajectory which is also the target
        # point by design for this test.
        config["target_radius"] = 0.072
        config["reward_scale"] = 2.0
        config["make_denser"] = False

        # Test : sparse reward
        # The last 3 steps are taken within the target radius, so the reward is 1*2.0
        # The rest of the test is the same as the dense case above.
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(20):
            # action = env.action_space.sample()
            action = np.array([0.5] * 2, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print(i, "sars', done =", state, action, reward, next_state, done)
            if i < 17:
                np.testing.assert_allclose(
                    0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            else:
                np.testing.assert_allclose(
                    2.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([0.69422, 1.27494]), atol=1e-5)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

        # Test delay
        # Adds delay to the previous test case
        config["delay"] = 10
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(35):
            # action = env.action_space.sample()
            action = np.array([0.5] * 2, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print(i, "sars', done =", state, action, reward, next_state, done)
            if i < 27 or i > 31:
                np.testing.assert_allclose(
                    0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            elif i >= 27 and i <= 31:
                np.testing.assert_allclose(
                    2.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            state = next_state.copy()
        np.testing.assert_allclose(state, np.array([1.06922, 1.64994]), atol=1e-5)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

        # Test irrelevant dimensions
        config["state_space_dim"] = 5
        config["action_space_dim"] = 5
        config["relevant_indices"] = [1, 2]
        config["action_space_relevant_indices"] = [1, 2]
        config["target_point"] = [1.27494, -0.780999]
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"].copy()  # env.reset()[0]
        for i in range(35):
            # action = env.action_space.sample()
            action = np.array([0.5] * 5, dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            if i < 27 or i > 31:
                # At each step, the distance reduces by ~0.035355 to the final point of
                # this trajectory which is also the target point by design for this test.
                # That is also the reward given at each step.
                np.testing.assert_allclose(
                    0.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            elif i >= 27 and i <= 31:
                np.testing.assert_allclose(
                    2.0, reward, atol=1e-5, err_msg="Step: " + str(i)
                )
            state = next_state.copy()
        np.testing.assert_allclose(
            state, np.array([1.06922, 1.64994, -0.405999, 1.89898, 0.0632061]), atol=1e-5
        )
        env.reset()[0]
        env.close()

    def test_continuous_image_representations(self):
        """"""
        print("\033[32;1;4mTEST_CONTINUOUS_IMAGE_REPRESENTATIONS\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = 0

        config["state_space_type"] = "continuous"
        config["action_space_type"] = "continuous"
        config["state_space_dim"] = 2
        config["action_space_dim"] = 2
        config["delay"] = 0
        config[
            "sequence_length"
        ] = 1  # seq_len is always going to be 1 for move_to_a_point R. assert for this?
        config["transition_dynamics_order"] = 1
        config["inertia"] = 1.0
        config["time_unit"] = 1

        config["reward_function"] = "move_to_a_point"
        # config["make_denser"] = False
        config["state_space_max"] = 5  # Will be a Box in the range [-max, max]
        config["target_point"] = [0.146517, -0.397534]
        config["target_radius"] = 0.172
        config["reward_scale"] = 2.0
        config["make_denser"] = False


        # Test 1: Sparse reward case; based on the initial state that is printed out
        # and the actions below, I set the target point to be what it is. Can optionally
        # uncomment the print statements to display the states.
        config["image_representations"] = True
        config["image_width"] = 100
        config["image_height"] = 100
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["augmented_state"][-1]
        # init state: [ 1.9652315 -2.4397445]
        expected_image_sums = [6168414, 6168414, 6168414, 6171735, 6204207]

        # obs = env.curr_obs[0]
        # import PIL.Image as Image
        # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
        # img1.show()

        for i in range(5):
            # action = env.action_space.sample()
            action = np.array([-0.45, -0.8], dtype=env.action_space.dtype)  # just to test if acting "in a line" works
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()

            # obs = env.curr_obs
            # import PIL.Image as Image
            # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
            # img1.show()

            if i < len(expected_image_sums):
                ...
                print(next_obs)
                assert next_obs.sum() == expected_image_sums[i], (
                    "Expected sum over image pixels: "
                    + str(expected_image_sums[i])
                    + ". Was: "
                    + str(next_obs.sum())
                )

        final_dist = np.linalg.norm(state - np.array(config["target_point"]))
        assert final_dist < config["target_radius"]

        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

    def test_grid_image_representations(self):
        """"""
        print("\033[32;1;4mTEST_GRID_IMAGE_REPRESENTATIONS\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = 0

        config["state_space_type"] = "grid"
        config["grid_shape"] = (8, 8)
        config["delay"] = 0
        config["sequence_length"] = 1

        config["reward_function"] = "move_to_a_point"
        config["target_point"] = [5, 5]
        config["reward_scale"] = 2.0
        config["make_denser"] = False

        config["image_representations"] = True

        # Test 1: Sparse reward case
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["augmented_state"][-1]
        actions = [
            [0, 1],
            [-1, 0],
            [-1, 0],
            [1, 0],
            [0.5, -0.5],
            [1, 2],
            [1, 0],
            [0, -1],
            [0, -1],
        ]
        expected_image_sums = [6372018, 6371313, 6372018, 6371313, 6371313]

        # obs = env.curr_obs[0]
        # import PIL.Image as Image
        # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
        # img1.show()

        tot_rew = 0
        for i in range(len(actions)):
            # action = env.action_space.sample()
            action = actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

            # obs = env.curr_obs
            # import PIL.Image as Image
            # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
            # img1.show()

            if i < len(expected_image_sums):
                assert next_obs.sum() == expected_image_sums[i], (
                    "Expected sum over image pixels: "
                    + str(expected_image_sums[i])
                    + ". Was: "
                    + str(next_obs.sum())
                )

        # To check bouncing back behaviour of grid walls
        for i in range(4):
            # action = env.action_space.sample()
            action = [0, 1]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

        assert tot_rew == 2.0, str(tot_rew)
        assert state == [5, 7], str(state)
        # test_ = np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        # self.assertAlmostEqual(state, np.array([21.59339006, 20.68189965, 21.49608203, 20.19183292]), places=3) # Error
        env.reset()[0]
        env.close()

        # Test 2: Almost the same as above, but with make_denser
        config["make_denser"] = True
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["augmented_state"][-1]
        actions = [
            [0, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [0.5, -0.5],
            [1, 2],
            [-1, -1],
            [0, -1],
            [0, -1],
        ]

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

        assert tot_rew == 2.0, str(tot_rew)

        env.reset()[0]
        env.close()

        # Test 3: Almost the same as 2, but with terminal states and different actions
        # Once a terminal state is reached, the self.reached_terminal is True
        # and terminal reward is always handed out in addition to the current
        # reward.
        config["terminal_states"] = [[5, 5], [2, 3], [2, 4], [3, 3], [3, 4]]
        config["term_state_reward"] = -0.25

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["augmented_state"][-1]
        actions = [
            [0, 1],
            [-1, 0],
            [1, 0],
            [1, 0],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, 1],
            [-1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 0],
        ]

        # obs = env.curr_obs
        # import PIL.Image as Image
        # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
        # img1.show()

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

        assert tot_rew == -0.5, str(tot_rew)

        env.reset()[0]
        env.close()

        # Test 4: Almost the same as 3, but with irrelevant features
        config["irrelevant_features"] = True

        env = RLToyEnv(**config)
        state = env.curr_state
        actions = [
            [0, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [0.5, -0.5],
            [1, 2],
            [-1, -1],
            [0, -1],
            [0, -1],
        ]
        expected_image_sums = [12272400, 12271695]

        # obs = env.curr_obs[0]
        # import PIL.Image as Image
        # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
        # img1.show()

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i] + [0, 0]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.curr_state
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

            # obs = env.curr_obs
            # import PIL.Image as Image
            # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
            # img1.show()

            if i < len(expected_image_sums):
                assert next_obs.sum() == expected_image_sums[i], (
                    "Expected sum over image pixels: "
                    + str(expected_image_sums[i])
                    + ". Was: "
                    + str(next_obs.sum())
                )

        for i in range(len(actions)):
            action = [0, 0] + actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.curr_state
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

            # obs = env.curr_obs
            # import PIL.Image as Image
            # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
            # img1.show()


        assert tot_rew == -3, str(tot_rew)

        env.reset()[0]
        env.close()

        # Test 5: With transition noise
        # Only in steps 9 and 11 are noisy transitions observed with valid actions
        config["transition_noise"] = 0.5
        config["reward_scale"] = 1.0

        env = RLToyEnv(**config)
        state = env.curr_state
        actions = [
            [0, 1],
            [-1, 1],
            [-1, 0],
            [1, -1],
            [0.5, -0.5],
            [1, 2],
            [1, 1],
            [0, -1],
            [1, 0],
            [0, -1],
            [1, 0],
            [0, -1],
            [0, -1],
        ]

        # obs = env.curr_obs
        # import PIL.Image as Image
        # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
        # img1.show()

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i] + [0, 0]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.curr_state
            print(i, "sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

            # obs = env.curr_obs
            # import PIL.Image as Image
            # img1 = Image.fromarray(np.squeeze(obs), 'RGB')
            # img1.show()

        assert tot_rew == 0.75, str(tot_rew)

        env.reset()[0]
        env.close()

    def test_grid_env(self):
        """"""
        print("\033[32;1;4mTEST_GRID_ENV\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = 0

        config["state_space_type"] = "grid"
        config["grid_shape"] = (8, 8)
        config["delay"] = 0
        config["sequence_length"] = 1

        config["reward_function"] = "move_to_a_point"
        config["make_denser"] = True
        config["target_point"] = [5, 5]
        config["reward_scale"] = 3.0

        # Test 1: Copied from test 3 in test_grid_image_representations
        # But it seems reward scale is different and I test for expected rewards
        # at each time step.
        config["terminal_states"] = [[5, 5], [2, 3], [2, 4], [3, 3], [3, 4]]
        config["term_state_reward"] = -0.25
        env = RLToyEnv(**config)

        state = env.get_augmented_state()["augmented_state"][-1]
        actions = [
            [0, 1],
            [-1, 0],
            [1, 0],
            [1, 0],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, 1],
            [-1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 0],
        ]
        expected_rewards = [-1, -1, 1, -1, 1, 1, 1, 1, 0.75]
        for i in range(len(expected_rewards)):
            expected_rewards[i] = (expected_rewards[i]) * config["reward_scale"]

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when reward delay = "
                + str(config["delay"]),
            )
            state = next_state.copy()
            tot_rew += reward

        assert tot_rew == 8.25, str(tot_rew)

        env.reset()[0]
        env.close()

        # Test 2: Almost the same as 1, but with irrelevant features and no terminal reward
        config["irrelevant_features"] = True
        config["term_state_reward"] = 0.0

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["augmented_state"][-1]
        actions = [
            [0, -1],
            [-1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [-1, 0],
        ]
        expected_rewards = [-1, -1, 1, -1, 1, 1, 1, 1, 1]
        for i in range(len(expected_rewards)):
            expected_rewards[i] = (expected_rewards[i]) * config["reward_scale"]

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i] + [0, 0]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when reward delay = "
                + str(config["delay"]),
            )
            state = next_state.copy()
            tot_rew += reward

        # Perform actions only in irrelevant space and noop in relevant space
        for i in range(len(actions)):
            action = [0, 0] + actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state.copy()
            tot_rew += reward

        assert tot_rew == 9, str(tot_rew)

        env.reset()[0]
        env.close()

        # Test 3: Almost the same as 1, but with delay
        config["delay"] = 1
        config["irrelevant_features"] = False
        config["term_state_reward"] = -0.25
        env = RLToyEnv(**config)

        state = env.get_augmented_state()["augmented_state"][-1]
        actions = [
            [0, -1],
            [-1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [-1, 0],
            [0, 0],  # noop
            [0, 0],  # noop
        ]
        expected_rewards = [0, -1, -1, 1, -1, 1, 1, 1, 0.75, 0.75, -0.25]
        for i in range(len(expected_rewards)):
            expected_rewards[i] = (expected_rewards[i]) * config["reward_scale"]

        tot_rew = 0
        for i in range(len(actions)):
            action = actions[i]
            next_obs, reward, done, trunc, info = env.step(action)
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, action, reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when reward delay = "
                + str(config["delay"]),
            )
            state = next_state.copy()
            tot_rew += reward

        assert tot_rew == 6.75, str(tot_rew)

        env.reset()[0]
        env.close()

    def test_discrete_dynamics(self):
        """Tests the P dynamics. Tests whether actions taken in terminal states lead back to the same terminal state. Tests if state in discrete environments is an int."""
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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 3
        config["reward_scale"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]
        if type(state) != int:
            self.assertEqual(
                state.dtype, env.observation_space.dtype, "Type of discrete state should be: " + str(env.observation_space.dtype)
            )  # TODO Move this and the test_continuous_dynamics type checks to separate unit tests

        action = 2
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(
            next_state,
            4,
            "Mismatch in state expected by transition dynamics for step 1.",
        )
        state = next_state

        action = 4
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(
            next_state,
            2,
            "Mismatch in state expected by transition dynamics for step 2.",
        )
        state = next_state

        action = 0
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(
            next_state,
            5,
            "Mismatch in state expected by transition dynamics for step 3.",
        )
        self.assertEqual(
            done,
            True,
            "Mismatch in expectation that terminal state should have been reached by transition dynamics for step 3.",
        )
        state = next_state

        # Try a random action to see that terminal state leads back to same terminal state
        action = env.action_space.sample()
        next_state, reward, done, trunc, info = env.step(action)
        print("sars', done =", state, action, reward, next_state, done)
        self.assertEqual(
            next_state,
            state,
            "Mismatch in state expected by transition dynamics for step 4. Terminal state was reached in step 3 and any random action should lead back to same terminal state.",
        )
        state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_reward_delay(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_REWARD_DELAY\033[0m")
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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 3
        config["sequence_length"] = 1
        config["reward_scale"] = 1.0

        config["generate_random_mdp"] = True

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            6,
            2,
            5,
            4,
            5,
            2,
            3,
            np.random.default_rng().integers(config["action_space_size"]),
            4,
        ]  # 2nd last action is random just to check that last delayed reward works with any action
        expected_rewards = [0, 0, 0, 1, 0, 0, 0, 1, 0]
        expected_states = [5, 4, 0, 0, 5, 4, 6,]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when reward delay = 3.",
            )
            # self.assertEqual(state, expected_states[i], "Expected state mismatch in
            # time step: " + str(i + 1) + " when reward delay = 3.") # will not work
            # for 2nd last time step due to random action.
            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_rewardable_sequences(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_REWARDABLE_SEQUENCES\033[0m")
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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 3
        config["reward_every_n_steps"] = 1
        config["reward_scale"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            5,
            2,
            3,
            3,
            4,
            7,
            np.random.default_rng().integers(config["action_space_size"]),
            5,
        ]  #
        expected_rewards = [0, 0, 1, 0, 0, 1, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when sequence length = 3.",
            )
            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_p_noise(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_P_NOISE\033[0m")
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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 1
        config["reward_scale"] = 1.0
        config["transition_noise"] = 0.5

        config["generate_random_mdp"] = True
        # config["log_level"] = logging.INFO
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [6, 6, 2, np.random.default_rng().integers(config["action_space_size"])]  #
        expected_states = [
            1,
            5,
            4,
            1,
        ]  # Last state 3 is fixed for this test because of fixed seed for Env which selects the next noisy state.
        for i in range(len(actions)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                next_state,
                expected_states[i],
                "Expected next state mismatch in time step: "
                + str(i + 1)
                + " when P noise = 0.5.",
            )
            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_r_noise(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_R_NOISE\033[0m")
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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 1
        config["reward_scale"] = 1.0
        config["reward_noise"] = lambda s, a, rng: rng.normal(0, 0.5)

        config["generate_random_mdp"] = True
        config["log_level"] = logging.INFO
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [6, 6, 2, 1]  #
        expected_rewards = [
            1 + 0.32021,
            0.0524501,
            -0.267835,
            0.180798,
        ]  # 2nd state produces 'true' reward
        for i in range(len(actions)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: "
                + str(i + 1)
                + " when R noise = 0.5.",
            )

            state = next_state

        env.reset()[0]
        env.close()

    # TODO Test for make_denser; also one for creating multiple instances of an Env with the same config dict (can lead to issues because the dict is shared as I found with Ray's A3C imple.)
    # TODO Tests for imaginary rollouts for discrete and continuous - for different Ps and Rs

    def test_discrete_multiple_meta_features(self):
        """Tests using multiple dimensions of hardness together.

        #TODO Currently tests for seq, del and r noise, r scale, r shift together. Include others? Gets complicated with P noise: trying to avoid terminal states while still following a rewardable sequence. Maybe try low P noise to test this? Or low terminal state density?
        """
        print("\033[32;1;4mTEST_DISCRETE_MULTIPLE_META_FEATURES\033[0m")

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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 1
        config["sequence_length"] = 3
        config["reward_every_n_steps"] = 1
        config["reward_scale"] = 2.5
        config["reward_shift"] = -1.75
        # config["transition_noise"] = 0.1
        config["reward_noise"] = lambda s, a, rng: rng.normal(0, 0.5)

        config["generate_random_mdp"] = True
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            6,
            6,
            2,
            3,
            4,
            2,
            np.random.default_rng().integers(config["action_space_size"]),
            5,
        ]  #
        expected_rewards = [0, 0, 0, 1, 0, 1, 0, 0]
        expected_reward_noises = [
            -0.292808,
            0.770696,
            -1.01743611,
            -0.042768,
            0.78761320,
            -0.510087,
            -0.089978,
            0.48654863,
        ]
        for i in range(len(expected_rewards)):
            expected_rewards[i] = (
                expected_rewards[i] + expected_reward_noises[i]
            ) * config["reward_scale"] + config["reward_shift"]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: "
                + str(i + 1)
                + " when sequence length = 3, delay = 1.",
            )
            state = next_state

        env.reset()[0]
        env.close()

    # Commented out the following 2 tests after changing implementation of
    # irrelevant_features to not use MultiDiscrete and be much simpler

    # def test_discrete_multi_discrete(self):
    #     '''
    #     Same as the test_discrete_reward_delay test above except with state_space_size and action_space_size specified as vectors and the actions slightly different near the end.
    #     '''
    #     print('\033[32;1;4mTEST_DISCRETE_MULTI_DISCRETE\033[0m')
    #
    #     config = {}
    #     config["log_filename"] = log_filename
    #     config["seed"] = {}
    #     config["seed"]["env"] = 0
    #     config["seed"]["relevant_state_space"] = 8
    #     config["seed"]["relevant_action_space"] = 8
    #
    #     config["state_space_type"] = "discrete"
    #     config["action_space_type"] = "discrete"
    #     config["state_space_size"] = [2, 2, 2]
    #     config["relevant_indices"] = [0, 1, 2]
    #     config["action_space_size"] = [2, 2, 2]
    #     config["action_space_relevant_indices"] = [0, 1, 2]
    #     config["reward_density"] = 0.25
    #     config["make_denser"] = True
    #     config["terminal_state_density"] = 0.25
    #     config["maximally_connected"] = True
    #     config["repeats_in_sequences"] = False
    #     config["delay"] = 3
    #     config["sequence_length"] = 1
    #     config["reward_scale"] = 1.0
    #
    #     config["generate_random_mdp"] = True
    #
    #     env = RLToyEnv(**config)
    #     state = env.get_augmented_state()['curr_state']
    #
    #     actions = [[1, 1, 0], [0, 1, 0], [1, 0 ,1], [1, 0 ,0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]]
    #     expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
    #     for i in range(len(expected_rewards)):
    #         next_state, reward, done, trunc, info = env.step(actions[i])
    #         print("sars', done =", state, actions[i], reward, next_state, done)
    #         self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
    #         state = next_state
    #
    #     env.reset()[0]
    #     env.close()

    # def test_discrete_multi_discrete_irrelevant_dimensions(self):
    #     '''
    #     Same as the test_discrete_multi_discrete test above except with state_space_size and action_space_size having extra irrelevant dimensions
    #     '''
    #     print('\033[32;1;4mTEST_DISCRETE_MULTI_DISCRETE_IRRELEVANT_DIMENSIONS\033[0m')
    #
    #     config = {}
    #     config["log_filename"] = log_filename
    #     config["seed"] = {}
    #     config["seed"]["env"] = 0
    #     config["seed"]["relevant_state_space"] = 8
    #     config["seed"]["relevant_action_space"] = 8
    #     config["seed"]["irrelevant_state_space"] = 52
    #     config["seed"]["irrelevant_action_space"] = 65
    #     config["seed"]["state_space"] = 87
    #     config["seed"]["action_space"] = 89
    #
    #     config["state_space_type"] = "discrete"
    #     config["action_space_type"] = "discrete"
    #     config["state_space_size"] = [2, 2, 2, 3]
    #     config["relevant_indices"] = [0, 1, 2]
    #     config["action_space_size"] = [2, 5, 2, 2]
    #     config["action_space_relevant_indices"] = [0, 2, 3]
    #     config["reward_density"] = 0.25
    #     config["make_denser"] = True
    #     config["terminal_state_density"] = 0.25
    #     config["maximally_connected"] = True
    #     config["repeats_in_sequences"] = False
    #     config["delay"] = 3
    #     config["sequence_length"] = 1
    #     config["reward_scale"] = 1.0
    #
    #     config["generate_random_mdp"] = True
    #
    #     try: # Testing for maximally_connected options working properly when invalid config specified. #TODO Is this part needed?
    #         env = RLToyEnv(**config)
    #         state = env.get_augmented_state()['curr_state']
    #
    #         actions = [[1, 1, 0], [0, 1, 0], [1, 0 ,1], [1, 0 ,0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]]
    #         expected_rewards = [0, 0, 0, 0, 1, 1, 0, 1, 0]
    #         for i in range(len(expected_rewards)):
    #             next_state, reward, done, trunc, info = env.step(actions[i])
    #             print("sars', done =", state, actions[i], reward, next_state, done)
    #             self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
    #             state = next_state
    #
    #         env.reset()[0]
    #         env.close()
    #
    #     except AssertionError as e:
    #         print('Caught Expected exception:', e)
    #
    #
    #     # Test: Adds one irrelevant dimension
    #     config["state_space_size"] = [2, 2, 2, 5]
    #     env = RLToyEnv(**config)
    #     state = env.get_augmented_state()['curr_state']
    #
    #     actions = [[1, 4, 1, 0], [0, 3, 1, 0], [1, 4, 0, 1], [1, 0 ,0, 0], [1, 2, 0, 1], [0, 3, 1, 0], [0, 1, 1, 1], [0, 4, 0, 1], [1, 4, 0, 0]]
    #     expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
    #     expected_states = [[0, 0, 0, 3], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 3], [0, 1, 0, 2], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 0, 4], [1, 0, 0, 2]]
    #     for i in range(len(expected_rewards)):
    #         next_state, reward, done, trunc, info = env.step(actions[i])
    #         print("sars', done =", state, actions[i], reward, next_state, done)
    #         self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
    #         self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
    #         state = next_state
    #
    #     env.reset()[0]
    #     env.close()

    # Test: This test lets even irrelevant dimensions be multi-dimensional
    # config["state_space_size"] = [2, 2, 2, 1, 5]
    # config["relevant_indices"] = [0, 1, 2]
    # config["action_space_size"] = [2, 5, 1, 1, 2, 2]
    # config["action_space_relevant_indices"] = [0, 4, 5]
    # env = RLToyEnv(**config)
    # state = env.get_augmented_state()['curr_state']
    #
    # actions = [[1, 4, 0, 0, 1, 0], [0, 3, 0, 0, 1, 0], [1, 4, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 1], [0, 3, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 4, 0, 0, 0, 1], [1, 4, 0, 0, 0, 0]]
    # expected_rewards = [0, 0, 0, 1, 1, 0, 1, 0, 0]
    # expected_states = [[0, 0, 0, 0, 3], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 0, 3], [0, 1, 0, 0, 2], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 4], [1, 0, 0, 0, 2]]
    # for i in range(len(expected_rewards)):
    #     next_state, reward, done, trunc, info = env.step(actions[i])
    #     print("sars', done =", state, actions[i], reward, next_state, done)
    #     self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
    #     self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = 3.")
    #     state = next_state
    #
    # env.reset()[0]
    # env.close()

    def test_discrete_irr_features(self):
        """ """
        print("\033[32;1;4mTEST_DISCRETE_IRR_FEATURES\033[0m")

        config = {}
        config["log_filename"] = log_filename
        config["seed"] = 0

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = [8, 10]
        config["action_space_size"] = [8, 10]
        config["irrelevant_features"] = True
        config["reward_density"] = 0.25
        config["make_denser"] = True
        config["terminal_state_density"] = 0.25
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 1
        config["sequence_length"] = 1
        config["reward_scale"] = 1.0

        config["generate_random_mdp"] = True

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [[7, 0], [5, 0], [5, 0], [1, 2]] + [
            [5, np.random.default_rng().integers(config["action_space_size"][1])]
        ] * 5
        expected_rewards = [0, 1, 0, 1, 0, 0, 0, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when reward delay = "
                + str(config["delay"]),
            )
            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_image_representations(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_IMAGE_REPRESENTATIONS\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = {}
        config["seed"]["env"] = 0
        config["seed"]["relevant_state_space"] = 8
        config["seed"]["relevant_action_space"] = 8
        config["seed"]["image_representations"] = 0

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.25
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 1
        config["sequence_length"] = 3
        config["reward_every_n_steps"] = 1
        config["reward_scale"] = 2.5
        config["reward_shift"] = -1.75
        # config["transition_noise"] = 0.1
        config["reward_noise"] = lambda s, a, rng: rng.normal(0, 0.5)

        config["generate_random_mdp"] = True

        config["image_representations"] = True
        config["image_width"] = 100
        config["image_height"] = 100
        config["image_transforms"] = "shift,scale,rotate,flip"
        config["image_scale_range"] = (0.5, 1.5)
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["augmented_state"][-1]

        actions = [
            6,
            6,
            2,
            3,
            4,
            2,
            np.random.default_rng().integers(config["action_space_size"]),
            5,
        ]  #
        expected_rewards = [0, 0, 0, 1, 0, 1, 0, 0]
        expected_reward_noises = [
            -0.292808,
            0.770696,
            -1.01743611,
            -0.042768,
            0.78761320,
            -0.510087,
            -0.089978,
            0.48654863,
        ]
        expected_image_sums = [
            122910,
            212925,
            111180,
        ]  # [152745, 282030, 528870], [105060, 232050, 78795]
        for i in range(len(expected_rewards)):
            expected_rewards[i] = (
                expected_rewards[i] + expected_reward_noises[i]
            ) * config["reward_scale"] + config["reward_shift"]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            assert next_state.shape == (
                100,
                100,
                1,
            ), "Expected shape was (100, 100, 1). Shape was:" + str(next_state.shape)
            assert (
                next_state.dtype == np.uint8
            ), "Expected dtype: np.uint8. Was: " + str(next_state.dtype)
            # import PIL.Image as Image
            # img1 = Image.fromarray(np.squeeze(next_state), 'L')
            # img1.show()
            if i < len(expected_image_sums):
                assert next_state.sum() == expected_image_sums[i], (
                    "Expected sum over image pixels: "
                    + str(expected_image_sums[i])
                    + ". Was: "
                    + str(next_state.sum())
                )  # Rotation changes the expected sum of 255 * 10201 = 2601255
            next_state = env.get_augmented_state()["augmented_state"][-1]
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: "
                + str(i + 1)
                + " when sequence length = 3, delay = 1.",
            )
            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_reward_every_n_steps(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_REWARD_EVERY_N_STEPS\033[0m")
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
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 3
        config["reward_scale"] = 1.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            6,
            6,
            2,
            3,
            4,
            2,
            6,
            1,
            0,
            np.random.default_rng().integers(config["action_space_size"]),
            5,
        ]  #
        expected_rewards = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when sequence length = 3.",
            )
            state = next_state

        env.reset()[0]
        env.close()

        # With delay
        config["delay"] = 1
        config["sequence_length"] = 3

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            6,
            6,
            2,
            3,
            4,
            2,
            6,
            1,
            0,
            np.random.default_rng().integers(config["action_space_size"]),
            5,
        ]  #
        expected_rewards = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when sequence length = 3.",
            )
            state = next_state

        env.reset()[0]
        env.close()

        # With delay >= sequence length
        config["delay"] = 1
        config["sequence_length"] = 1

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            6,
            6,
            2,
            3,
            4,
            2,
            6,
            1,
            0,
            np.random.default_rng().integers(config["action_space_size"]),
            5,
        ]  #
        expected_rewards = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: "
                + str(i + 1)
                + " when sequence length = 1.",
            )
            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_custom_P_R(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_CUSTOM_P_R\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = 0

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 5
        config["terminal_state_density"] = 0.25
        # config["maximally_connected"] = False
        config["repeats_in_sequences"] = False
        config["delay"] = 1
        config["reward_scale"] = 2.0

        config["use_custom_mdp"] = True
        # np.random.seed(0)  # seed
        config["transition_function"] = np.random.default_rng().integers(8, size=(8, 5))
        config["reward_function"] = np.random.default_rng().integers(4, size=(8, 5))
        config["init_state_dist"] = np.array([1 / 8 for i in range(8)])

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            4,
            4,
            2,
            3,
            4,
            2,
            4,
            1,
            0,
            np.random.default_rng(0).integers(config["action_space_size"]),
            4,
        ]  #
        expected_rewards = [0, 0, 6, 4, 4, 0, 4, 6, 6, 2, 0]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: " + str(i + 1) + ".",
            )
            state = next_state

        env.reset()[0]
        env.close()

        # np.random.seed(0) #seed
        config["delay"] = 2
        P = np.random.default_rng().integers(8, size=(8, 5))
        R = np.random.default_rng().integers(4, size=(8, 5))
        config["transition_function"] = lambda s, a: P[s, a]
        config["reward_function"] = lambda s, a: R[s[-2], a]
        config["init_state_dist"] = np.array([1 / 8 for i in range(8)])

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [
            4,
            4,
            2,
            3,
            4,
            2,
            4,
            1,
            0,
            np.random.default_rng().integers(config["action_space_size"]),
            4,
        ]  #
        expected_rewards = [0, 0, 0, 2, 2, 0, 0, 6, 0, 4, 6]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            self.assertEqual(
                reward,
                expected_rewards[i],
                "Expected reward mismatch in time step: " + str(i + 1) + ".",
            )
            state = next_state

        env.reset()[0]
        env.close()

    def test_continuous_custom_P_R(self):
        """"""
        print("\033[32;1;4mTEST_CONT_CUSTOM_P_R\033[0")
        config = {}
        config["log_filename"] = log_filename
        config["seed"] = 0

        config["state_space_type"] = "continuous"
        config["action_space_type"] = "continuous"
        config["state_space_dim"] = 2
        config["action_space_dim"] = 2
        # config["target_point"] = [0, 0]
        config["reward_scale"] = 1.0
        config["delay"] = 1

        config["use_custom_mdp"] = True
        # np.random.seed(0)  # seed
        config["transition_function"] = lambda s, a: s + a
        config["reward_function"] = lambda s, a: s[-2][0]
        # config["init_state_dist"] = np.array([1 / 8 for i in range(8)])

        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [2, [0.5, 1.5], 2, 3, [-10, -5], 2, 1, 1]  #
        expected_rewards = [
            0,
            -1.06496,
            0.935036,
            1.435036,
            3.435036,
            6.435036,
            -3.564964,
            -1.564964,
        ]  # , -0.564964]
        for i in range(len(expected_rewards)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: " + str(i + 1) + ".",
            )
            state = next_state

        env.reset()[0]
        env.close()

    # def test_discrete_imaginary_rollouts(self):
    #     '''### TODO complete test case
    #     '''
    #     print('\033[32;1;4mTEST_DISCRETE_IMAGINARY_ROLLOUTS\033[0m')
    #
    #     config = {}
    #     config["log_filename"] = log_filename
    #     # config["log_level"] = logging.NOTSET
    #     config["seed"] = 0
    #
    #     config["state_space_type"] = "discrete"
    #     config["action_space_type"] = "discrete"
    #     config["state_space_size"] = 20
    #     config["action_space_size"] = 20
    #     config["reward_density"] = 1e-4 # 160392960 possible sequences for 18 non-terminal states and seq_len 7; 1028160 for seq_len 5
    #     config["make_denser"] = False
    #     config["terminal_state_density"] = 0.1
    #     config["delay"] = 2
    #     config["sequence_length"] = 5
    #     config["reward_scale"] = 1.0
    #     config["maximally_connected"] = True
    #     config["repeats_in_sequences"] = False
    #
    #     config["generate_random_mdp"] = True
    #
    #     env = RLToyEnv(**config)
    #     state = env.get_augmented_state()['curr_state']
    #
    #     actions = [0, 1, 17, 5, 3, 4, 3, 2]
    #     expected_rewards = [0, 0, 0, 0, 0, 0]#, 1, 0, 0]
    #     expected_states = [9, 2, 4, 5, 8, 9] # [2, 4, 5, 8, 9] is a rewardable sequence. init state is 9 and action 0 leads to state 2.
    #     for i in range(len(expected_rewards)):
    #         next_state, reward, done, trunc, info = env.step(actions[i])
    #         print("sars', done =", state, actions[i], reward, next_state, done)
    #         self.assertEqual(reward, expected_rewards[i], "Expected reward mismatch in time step: " + str(i + 1) + " when reward delay = " + str(config["delay"]))
    #         self.assertEqual(state, expected_states[i], "Expected state mismatch in time step: " + str(i + 1) + " when reward delay = " + str(config["delay"]))
    #         state = next_state
    #
    #     env.reset()[0]
    #     env.close()

    def test_discrete_r_dist(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_R_DIST\033[0m")
        config = {}
        config["log_filename"] = log_filename
        # config["log_level"] = logging.NOTSET
        config["seed"] = 0

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 8
        config["action_space_size"] = 8
        config["reward_density"] = 0.5
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["sequence_length"] = 1
        config["reward_scale"] = 1.0
        config["reward_shift"] = 1.0
        config["reward_dist"] = lambda rng, r_dict: rng.normal(0, 0.5)

        config["generate_random_mdp"] = True
        env = RLToyEnv(**config)
        state = env.get_augmented_state()["curr_state"]

        actions = [6, 6, 2, 6]  #
        expected_rewards = [
            1.131635,
            1,
            0.316987,
            1.424395,
        ]  # 1st, 3rd and 4th states produce 'true' rewards, every reward has been shifted by 1
        for i in range(len(actions)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: "
                + str(i + 1)
                + " when R dist = 0.5.",
            )

            state = next_state

        env.reset()[0]
        env.close()

    def test_discrete_diameter(self):
        """"""
        print("\033[32;1;4mTEST_DISCRETE_DIAMETER\033[0m")
        config = {}
        config["log_filename"] = log_filename
        config["log_level"] = logging.NOTSET
        config["seed"] = 0

        config["state_space_type"] = "discrete"
        config["action_space_type"] = "discrete"
        config["state_space_size"] = 24
        config["action_space_size"] = 8
        config["reward_density"] = 0.05
        config["make_denser"] = False
        config["terminal_state_density"] = 0.25
        config["maximally_connected"] = True
        config["repeats_in_sequences"] = False
        config["delay"] = 0
        config["diameter"] = 3
        config["sequence_length"] = 3
        config["reward_every_n_steps"] = 1
        config["reward_scale"] = 1.0
        config["reward_shift"] = 0.0

        config["generate_random_mdp"] = True
        env = RLToyEnv(**config)

        for seq_num, sequence in enumerate(env.rewardable_sequences):
            for i, state in enumerate(sequence):
                assert state not in [
                    6,
                    7,
                    14,
                    15,
                    22,
                    23,
                ], "A terminal state was encountered in a rewardable sequence. This is unexpected."
                rem_ = state % env.action_space_size[0]
                assert rem_ < env.action_space_size[0] - env.num_terminal_states, (
                    "The effective state number within an independent set was expected to be in range (0, 6). However, it was: "
                    + str(rem_)
                )

        assert len(env.rewardable_sequences) == int(0.05 * np.prod([6, 6, 6])) * 3, (
            "Number of rewardable_sequences: "
            + str(len(env.rewardable_sequences))
            + ". Expected: "
            + str(int(0.05 * np.prod([6, 6, 6])) * 3)
        )
        np.testing.assert_allclose(
            np.sum(env.config["relevant_init_state_dist"]),
            1.0,
            rtol=1e-05,
            err_msg="Expected sum of probabilities in init_state_dist was 1.0. However, actual sum was: "
            + str(np.sum(env.config["relevant_init_state_dist"])),
        )  # TODO Similar test case for irrelevant_features

        state = env.get_augmented_state()["curr_state"]
        actions = [6, 6, 7, 7, 0, 7, 1]  #
        expected_rewards = [
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ]  # 1st, 3rd and 4th states produce 'true' rewards, every reward has been shifted by 1
        for i in range(len(actions)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: "
                + str(i + 1)
                + " when diam = "
                + str(config["diameter"]),
            )

            state = next_state

        env.reset()[0]
        env.close()

        # Sub-test 2 Have sequence length greater than the diameter and check selected rewardable sequences
        config["sequence_length"] = 5
        config[
            "reward_density"
        ] = 0.01  # reduce density to have fewer rewardable sequences

        env = RLToyEnv(**config)

        for seq_num, sequence in enumerate(env.rewardable_sequences):
            for j in range(config["diameter"]):
                if (
                    j / config["diameter"] < seq_num / len(env.rewardable_sequences)
                    and seq_num / len(env.rewardable_sequences)
                    < (j + 1) / config["diameter"]
                ):
                    for i, state in enumerate(sequence):
                        min_state = (
                            (i + j) * env.action_space_size[0]
                        ) % env.state_space_size[0]
                        max_state = (
                            (i + j + 1) * env.action_space_size[0]
                        ) % env.state_space_size[0]
                        if max_state < min_state:  # edge case
                            max_state += env.state_space_size[0]
                        assert sequence[i] >= min_state and sequence[i] < max_state, (
                            ""
                            + str(min_state)
                            + " "
                            + str(sequence[i])
                            + " "
                            + str(max_state)
                        )

            for i, state in enumerate(sequence):
                rem_ = state % env.action_space_size[0]
                assert rem_ < env.action_space_size[0] - env.num_terminal_states, (
                    "The effective state number within an independent set was expected to be in range (0, 6). However, it was: "
                    + str(rem_)
                )

        assert (
            len(env.rewardable_sequences) == int(0.01 * np.prod([6, 6, 6, 5, 5])) * 3
        ), (
            "Number of rewardable_sequences: "
            + str(len(env.rewardable_sequences))
            + ". Expected: "
            + str(int(0.05 * np.prod([6, 6, 6, 5, 5])) * 3)
        )
        np.testing.assert_allclose(
            np.sum(env.config["relevant_init_state_dist"]),
            1.0,
            rtol=1e-05,
            err_msg="Expected sum of probabilities in init_state_dist was 1.0. However, actual sum was: "
            + str(np.sum(env.config["relevant_init_state_dist"])),
        )  # TODO Similar test case for irrelevant_features

        state = env.get_augmented_state()["curr_state"]
        actions = [1, 7, 2, 4, 0, 7, 1]  # Leads to rewardable sequence 20, 1, 12, 21, 5
        expected_rewards = [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
        ]  # 1st, 3rd and 4th states produce 'true' rewards, every reward has been shifted by 1
        for i in range(len(actions)):
            next_state, reward, done, trunc, info = env.step(actions[i])
            print("sars', done =", state, actions[i], reward, next_state, done)
            np.testing.assert_allclose(
                reward,
                expected_rewards[i],
                rtol=1e-05,
                err_msg="Expected reward mismatch in time step: "
                + str(i + 1)
                + " when diam = "
                + str(config["diameter"]),
            )

            state = next_state

        env.reset()[0]
        env.close()

    # Unit tests


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Need to set log level for RLToyEnv in the config
    # of each test and setting it here actually doesn't work for the logging stuff from RLToyEnv.
    rl_toy = TestRLToyEnv()
    # rl_toy.setUp()
    # rl_toy.test_discrete_rewardable_sequences()
    # rl_toy.test_discrete_p_noise()
    rl_toy.test_discrete_r_noise()
    # unittest.main()
