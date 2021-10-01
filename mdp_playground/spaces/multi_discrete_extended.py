import numpy as np
import gym
from gym.spaces import MultiDiscrete


class MultiDiscreteExtended(MultiDiscrete):
    """
    Currently, this class doesn't do anything special except allowing to seed from __init__().
    """

    def __init__(self, nvec, seed=None):
        super(MultiDiscreteExtended, self).__init__(nvec)
        super(MultiDiscreteExtended, self).seed(seed=seed)

    # def sample(self, prob=None, size=1, replace=True):
    #     sampled = np.squeeze(self.np_random.choice(self.n, size=size, p=prob, replace=replace))
    #     if sampled.shape == ():
    #         sampled = int(sampled) #TODO Just made it an int otherwise np.array(scalar) looks ugly in output.
    #     return sampled
