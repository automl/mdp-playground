import numpy as np
import gym
from gym.spaces import Discrete, Box

class DiscreteExtended(Discrete):

    def sample(self, prob=None, size=1, replace=True):
        return self.np_random.choice(self.n, size=size, p=prob, replace=replace)
