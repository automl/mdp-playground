import numpy as np
import gym
from mdp_playground.spaces import BoxExtended


class GridActionSpace(BoxExtended):
    def __init__(self, low, high, shape=None, seed=None):
        super(GridActionSpace, self).__init__(
            low, high, shape=shape, dtype=np.int64, seed=seed
        )
        assert len(self.shape) == 1

    def sample(self):
        samp = np.zeros(shape=self.high.shape)
        # Select which dimension will have action (only 1 dimension can have
        # motion in traditional grid worlds). This also is more consistent with
        # Manhattan dist reward defined for grid worlds in rl_toy_env.py
        ind = self.np_random.randint(self.high.size)
        val = self.np_random.randint(3)
        samp[ind] = val - 1  # Shift into grid action range of [-1, 0, 1]

        return samp.astype(int)

    def contains(self, x):
        x = np.array(x)
        if x.dtype.kind != "i":
            return False

        a = x == 0
        b = x == 1
        c = x == -1
        truth = a + b + c
        if not np.all(truth):
            return False

        if np.sum(np.abs(x)) != 0 and np.sum(np.abs(x)) != 1:
            return False

        return True
