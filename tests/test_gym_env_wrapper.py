import logging
import copy
import numpy as np
from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
import unittest
import pytest

import sys

# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, "./mdp_playground/envs/")
# sys.path.append('./mdp_playground/envs/')


# TODO logging
# from datetime import datetime
# log_filename = '/tmp/test_mdp_playground_' +
# datetime.today().strftime('%m.%d.%Y_%I:%M:%S_%f') + '.log' #TODO Make a
# directoy 'log/' and store there.


# TODO None of the tests do anything when done = True. Should try calling reset() in one of them and see that this works?


class TestGymEnvWrapper(unittest.TestCase):
    def test_r_delay(self):
        """ """
        print("\033[32;1;4mTEST_REWARD_DELAY\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "delay": 1,
            # "GymEnvWrapper": {
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv

        ae = AtariEnv(**{"game": "beam_rider", "obs_type": "image", "frameskip": 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print("step, reward, done, act:", i, reward, done, act)
            if i == 154 or i == 159:
                assert reward == 44.0, (
                    "1-step delayed reward in step: "
                    + str(i)
                    + " should have been 44.0."
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    def test_r_shift(self):
        """ """
        print("\033[32;1;4mTEST_REWARD_SHIFT\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "reward_shift": 1,
            # "GymEnvWrapper": {
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv

        ae = AtariEnv(**{"game": "beam_rider", "obs_type": "image", "frameskip": 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print("step, reward, done, act:", i, reward, done, act)
            if i == 153 or i == 158:
                assert reward == 45.0, (
                    "Shifted reward in step: " + str(i) + " should have been 45.0."
                )
            if i == 154 or i == 160:
                assert reward == 1.0, (
                    "Shifted reward in step: " + str(i) + " should have been 1.0."
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    def test_r_scale(self):
        """ """
        print("\033[32;1;4mTEST_REWARD_SHIFT\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "reward_scale": 2,
            # "GymEnvWrapper": {
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv

        ae = AtariEnv(**{"game": "beam_rider", "obs_type": "image", "frameskip": 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print("step, reward, done, act:", i, reward, done, act)
            if i == 153 or i == 158:
                assert reward == 88.0, (
                    "Scaled reward in step: " + str(i) + " should have been 88.0."
                )
            if i == 154 or i == 160:
                assert reward == 0.0, (
                    "Scaled reward in step: " + str(i) + " should have been 0.0."
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    def test_r_delay_ray_frame_stack(self):
        """
        Uses wrap_deepmind_ray to frame stack Atari
        """
        print("\033[32;1;4mTEST_REWARD_DELAY_RAY_FRAME_STACK\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "delay": 1,
            # "GymEnvWrapper": {
            "wrap_deepmind_ray": True,
            "frame_skip": 1,
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv
        import gym

        game = "beam_rider"
        game = "".join([g.capitalize() for g in game.split("_")])
        ae = gym.make("{}NoFrameskip-v4".format(game))
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print("step, reward, done, act:", i, reward, done, act)
            if i == 142 or i == 159:
                assert reward == 44.0, (
                    "1-step delayed reward in step: "
                    + str(i)
                    + " should have been 44.0."
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    def test_r_delay_p_noise_r_noise(self):
        """
        P noise is currently only for discrete env #TODO
        """
        print("\033[32;1;4mTEST_MULTIPLE\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "delay": 1,
            "reward_noise": lambda a: a.normal(0, 0.1),
            "transition_noise": 0.1,
            # "GymEnvWrapper": {
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv

        ae = AtariEnv(**{"game": "beam_rider", "obs_type": "image", "frameskip": 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print("step, reward, done, act:", i, reward, done, act)
            # Testing hardcoded values at these timesteps implicitly tests that there
            # were 21 noisy transitions in total and noise inserted in rewards.
            if i == 154:
                np.testing.assert_allclose(
                    reward,
                    44.12183457980473,
                    rtol=1e-05,
                    err_msg="1-step delayed reward in step: "
                    + str(i)
                    + " should have been 44.0.",
                )
            if i == 199:
                np.testing.assert_allclose(
                    reward,
                    0.07467690634910334,
                    rtol=1e-05,
                    err_msg="1-step delayed reward in step: "
                    + str(i)
                    + " should have been 44.0.",
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    def test_discrete_irr_features(self):
        """ """
        print("\033[32;1;4mTEST_DISC_IRR_FEATURES\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "delay": 1,
            # "GymEnvWrapper": {
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            "irrelevant_features": {
                "state_space_type": "discrete",
                "action_space_type": "discrete",
                "state_space_size": 8,
                "action_space_size": 8,
                "completely_connected": True,
                "repeats_in_sequences": False,
                "generate_random_mdp": True,
                # TODO currently RLToyEnv needs to have at least 1 terminal state, allow it to have 0 in future.
                "terminal_state_density": 0.2,
            }
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv

        ae = AtariEnv(**{"game": "beam_rider", "obs_type": "image", "frameskip": 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("type(observation_space):", type(ob))
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print(
                "step, reward, done, act, next_state[1]:",
                i,
                reward,
                done,
                act,
                next_state[1],
            )
            if i == 154 or i == 159:
                assert reward == 44.0, (
                    "1-step delayed reward in step: "
                    + str(i)
                    + " should have been 44.0."
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    def test_image_transforms(self):
        """ """
        print("\033[32;1;4mTEST_IMAGE_TRANSFORMS\033[0m")
        config = {
            "AtariEnv": {
                "game": "beam_rider",  # "breakout",
                "obs_type": "image",
                "frameskip": 1,
            },
            "image_transforms": "shift,scale,rotate",
            # "image_sh_quant": 2,
            "image_width": 40,
            "image_padding": 30,
            # "GymEnvWrapper": {
            "atari_preprocessing": True,
            "frame_skip": 4,
            "grayscale_obs": False,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "seed": 0,
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv

        ae = AtariEnv(**{"game": "beam_rider", "obs_type": "image", "frameskip": 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        assert ob.shape == (100, 100, 3), "Observation shape of the env was unexpected."
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print("step, reward, done, act:", i, reward, done, act)
            if i == 153 or i == 158:
                assert reward == 44.0, (
                    "Reward in step: " + str(i) + " should have been 44.0."
                )
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()

    @pytest.mark.skip(reason="Cannot run mojoco in CI/CD currently.")
    def test_cont_irr_features(self):
        """ """
        print("\033[32;1;4mTEST_CONT_IRR_FEATURES\033[0m")
        config = {
            # "AtariEnv": {
            #     "game": 'beam_rider', #"breakout",
            #     'obs_type': 'image',
            #     'frameskip': 1,
            # },
            # 'delay': 1,
            # "GymEnvWrapper": {
            "state_space_type": "continuous",
            "action_space_type": "continuous",
            "seed": 0,
            "irrelevant_features": {
                "state_space_type": "continuous",
                "action_space_type": "continuous",
                "state_space_dim": 2,
                "action_space_dim": 2,
                "reward_function": "move_to_a_point",
                "target_point": [3, 3],
                # "state_space_max": 10,
                # "action_space_max": 10,
                "transition_dynamics_order": 1,
                "dtype": np.float64
                # "generate_random_mdp": True,
                # 'terminal_state_density': 0.2, ##TODO currently RLToyEnv needs to have at least 1 terminal state, allow it to have 0 in future.
            }
            # },
            # 'seed': 0, #seed
        }

        # config["log_filename"] = log_filename

        from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper  # hack
        from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

        HalfCheetahWrapperV3 = get_mujoco_wrapper(HalfCheetahEnv)
        base_env_config = {}
        hc3 = HalfCheetahWrapperV3(**base_env_config)
        # register_env("HalfCheetahWrapper-v3", lambda config: HalfCheetahWrapperV3(**config))

        hc3w = GymEnvWrapper(hc3, **config)
        ob = hc3w.reset()
        print("obs shape, type(observation_space):", ob.shape, type(ob))
        print("initial obs: ", ob)
        assert (
            ob.shape[0]
            == hc3w.env.observation_space.shape[0]
            + hc3w.irr_toy_env.observation_space.shape[0]
        ), (
            "Shapes of base cont. env, irrelevant toy env: "
            + str(hc3w.env.observation_space.shape)
            + str(hc3w.irr_toy_env.observation_space.shape)
        )
        total_reward = 0.0

        # check that random elements are correctly set in the observation and action spaces of component env's
        assert hc3w.env.observation_space.low[3] == hc3w.observation_space.low[3], (
            "Concatenation of observation spaces of components envs to external observation spaces was not done correctly. Compared values: "
            + str(hc3w.env.observation_space.low[3])
            + ", "
            + str(hc3w.observation_space.low[3])
        )
        assert (
            hc3w.irr_toy_env.action_space.high[1]
            == hc3w.action_space.high[hc3w.env.action_space.shape[0] + 1]
        ), (
            "Concatenation of action spaces of components envs to external action spaces was not done correctly. Compared values: "
            + str(hc3w.irr_toy_env.action_space.high[1])
            + ", "
            + str(hc3w.action_space.high[hc3w.env.action_space.shape[0] + 1])
        )

        for i in range(200):
            act = hc3w.action_space.sample()
            next_state, reward, done, info = hc3w.step(act)
            print(
                "step, reward, done, act, next_state:", i, reward, done, act, next_state
            )
            if i == 0:
                np.testing.assert_allclose(
                    next_state[hc3w.env.observation_space.shape[0] :],
                    [-4.51594779e-01, -1.00795288e00],
                    rtol=1e-05,
                    err_msg="Mismatch for irr. toy envs' state in step " + str(i) + ".",
                )
            elif i == 2:
                np.testing.assert_allclose(
                    next_state[hc3w.env.observation_space.shape[0] :],
                    [-2.95129723, 0.05893834],
                    rtol=1e-05,
                    err_msg="Mismatch for irr. toy envs' state in step " + str(i) + ".",
                )
            total_reward += reward
        print("total_reward:", total_reward)
        hc3w.reset()


if __name__ == "__main__":
    unittest.main()
