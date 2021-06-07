import numpy as np
import gym
from gym.spaces import Discrete


class DiscreteExtended(Discrete):
    def __init__(self, n, seed=None):
        super(DiscreteExtended, self).__init__(n)
        super(DiscreteExtended, self).seed(seed=seed)

    def sample(self, max=None, prob=None, size=1, replace=True):
        if (
            max is None
        ):  # hack This was done when diameter was introduced as a dimension so as not to have to change the tests which would fail.
            max = self.n
        sampled = np.squeeze(
            self.np_random.choice(max, size=size, p=prob, replace=replace)
        )
        if sampled.shape == ():
            sampled = int(
                sampled
            )  # TODO Just made it an int otherwise np.array(scalar) looks ugly in output.
        return sampled
