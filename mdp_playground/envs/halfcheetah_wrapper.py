from gym.envs.mujoco.halfcheetah_v3 import HalfCheetahEnv
import copy

class HalfCheetahWrapperV3(HalfCheetahEnv):
    def __init__(self, **config):
        action_space_max = config["action_space_max"]
        self.config = copy.deepcopy(config)
        if "dummy_eval" in config:
            del config["dummy_eval"]
        del config["action_space_max"]
        super(HalfCheetahWrapperV3, self).__init__(**config)
        self.action_space.low *= action_space_max
        self.action_space.high *= action_space_max
