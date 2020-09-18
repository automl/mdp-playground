import unittest

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './mdp_playground/envs/')
# sys.path.append('./mdp_playground/envs/')
from gym_env_wrapper import GymEnvWrapper

import numpy as np
import copy
import logging

##TODO logging
# from datetime import datetime
# log_filename = '/tmp/test_mdp_playground_' + datetime.today().strftime('%m.%d.%Y_%I:%M:%S_%f') + '.log' #TODO Make a directoy 'log/' and store there.


#TODO None of the tests do anything when done = True. Should try calling reset() in one of them and see that this works?

class TestGymEnvWrapper(unittest.TestCase):

    def test_delay(self):
        '''
        '''
        print('\033[32;1;4mTEST_DELAY\033[0m')
        config = {
                "AtariEnv": {
                    "game": 'beam_rider', #"breakout",
                    'obs_type': 'image',
                    'frameskip': 1,
                },
                'delay': 1,
                # "GymEnvWrapper": {
                "atari_preprocessing": True,
                'frame_skip': 4,
                'grayscale_obs': False,
                'state_space_type': 'discrete',
                'action_space_type': 'discrete',
                'seed': 0,
                # },
                # 'seed': 0, #seed
            }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv
        ae = AtariEnv(**{'game': 'beam_rider', 'obs_type': 'image', 'frameskip': 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print(reward, done, act)
            if i == 154 or i == 159:
                assert reward == 44.0, "1-step delayed reward in step: " + str(i) + " should have been 44.0."
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()


    def test_delay_p_noise_r_noise(self):
        '''
        '''
        print('\033[32;1;4mTEST_DELAY\033[0m')
        config = {
                "AtariEnv": {
                    "game": 'beam_rider', #"breakout",
                    'obs_type': 'image',
                    'frameskip': 1,
                },
                'delay': 1,
                'reward_noise': lambda a: a.normal(0, 0.1),
                'transition_noise': 0.1,
                # "GymEnvWrapper": {
                "atari_preprocessing": True,
                'frame_skip': 4,
                'grayscale_obs': False,
                'state_space_type': 'discrete',
                'action_space_type': 'discrete',
                'seed': 0,
                # },
                # 'seed': 0, #seed
            }

        # config["log_filename"] = log_filename

        from gym.envs.atari import AtariEnv
        ae = AtariEnv(**{'game': 'beam_rider', 'obs_type': 'image', 'frameskip': 1})
        aew = GymEnvWrapper(ae, **config)
        ob = aew.reset()
        print("observation_space.shape:", ob.shape)
        # print(ob)
        total_reward = 0.0
        for i in range(200):
            act = aew.action_space.sample()
            next_state, reward, done, info = aew.step(act)
            print(reward, done, act)
            # Testing hardcoded values at these timesteps implicitly tests that there were 21 noisy transitions in total and noise inserted in rewards.
            if i == 154:
                np.testing.assert_allclose(reward, 44.12183457980473, rtol=1e-05, err_msg="1-step delayed reward in step: " + str(i) + " should have been 44.0.")
            if i == 199:
                np.testing.assert_allclose(reward, 0.07467690634910334, rtol=1e-05, err_msg="1-step delayed reward in step: " + str(i) + " should have been 44.0.")
            total_reward += reward
        print("total_reward:", total_reward)
        aew.reset()



if __name__ == '__main__':
    unittest.main()
