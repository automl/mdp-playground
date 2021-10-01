import numpy as np
import gym
from gym.spaces import Box


class BoxExtended(Box):
    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        super(BoxExtended, self).__init__(low, high, shape=shape, dtype=dtype)
        super(BoxExtended, self).seed(seed=seed)

    # def sample(self, prob=None, size=1, replace=True):
    #     sampled = np.squeeze(self.np_random.choice(self.n, size=size, p=prob, replace=replace))
    #     if sampled.shape == ():
    #         sampled = int(sampled) #TODO Just made it an int otherwise np.array(scalar) looks ugly in output.
    #     return sampled
