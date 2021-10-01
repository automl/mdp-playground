import numpy as np
import gym
from gym.spaces import Tuple


class TupleExtended(Tuple):
    """
    Currently, this class doesn't do anything special except allowing to seed from __init__().
    """

    def __init__(self, spaces, seed=None):
        super(TupleExtended, self).__init__(spaces)
        super(TupleExtended, self).seed(seed=seed)

    # def sample(self, prob=None, size=1, replace=True):
    #     sampled = np.squeeze(self.np_random.choice(self.n, size=size, p=prob, replace=replace))
    #     if sampled.shape == ():
    #         sampled = int(sampled) #TODO Just made it an int otherwise np.array(scalar) looks ugly in output.
    #     return sampled
