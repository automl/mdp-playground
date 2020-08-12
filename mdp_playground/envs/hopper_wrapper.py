from gym.envs.mujoco.hopper_v3 import HopperEnv


class HopperWrapperV3(HopperEnv):
    def __init__(self, **config):
        action_space_max = config["action_space_max"]
        self.config = config
        if "dummy_eval" in config:
            del config["dummy_eval"]
        del config["action_space_max"]
        super(HopperWrapperV3, self).__init__(**config)
        self.action_space.low *= action_space_max
        self.action_space.high *= action_space_max

