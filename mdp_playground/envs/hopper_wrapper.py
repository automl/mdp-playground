from gym.envs.mujoco.hopper_v3 import HopperEnv


class HopperWrapperV3(HopperEnv):
    def __init__(self, **config):

        super(HopperWrapperV3, self).__init__(**config)
        self.action_space.low *= config["action_space_max"]
        self.action_space.high *= config["action_space_max"]
