import numpy as np
import gym
from gym.spaces import Discrete, Box

class DiscreteExtended(Discrete):

    def sample(self, prob=None, size=1, replace=True):
        sampled = np.squeeze(self.np_random.choice(self.n, size=size, p=prob, replace=replace))
        if sampled.shape == ():
            sampled = int(sampled) #TODO Just made it an int otherwise np.array(scalar) looks ugly in output
        return sampled
