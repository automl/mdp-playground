# Currently just a placeholder file I think
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box


class RLToyEnv(gym.Env):
    """Example of a custom env"""

    def __init__(self, config):
        self.max_real = 100.0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.max_real, shape=(1, ), dtype=np.float64)
        self.cur_state = 0

    def reset(self):
        self.cur_state = 0
        return [self.cur_state]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_state > 0:
            self.cur_state -= 1
        elif action == 1:
            self.cur_state += 1
        done = self.cur_state >= self.max_real
        return [self.cur_state], 1 if done else 0, done, {}


if __name__ == "__main__":

    env = RLToyEnv([])
    state = env.reset()
    for _ in range(1000):
        # env.render() # For GUI
        action = env.action_space.sample() # take a random action
        next_state, reward, done, info = env.step(action)
        print("sars =", state, action, reward, next_state)
        state = next_state
    env.close()
