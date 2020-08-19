from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import copy

class HalfCheetahWrapperV3(HalfCheetahEnv):
    def __init__(self, **config):
        action_space_max = config["action_space_max"]
        self.config = copy.deepcopy(config)
        if "dummy_eval" in config: #hack
            del config["dummy_eval"]
        if "transition_noise" in config: #hack
            del config["transition_noise"]
        if "reward_noise" in config: #hack
            del config["reward_noise"]
        if "action_loss_weight" in config: #hack
            del config["action_loss_weight"]
        if "action_space_max" in config: #hack
            del config["action_space_max"]
        if "dummy_seed" in config: #hack
            del config["dummy_seed"]

        super(HalfCheetahWrapperV3, self).__init__(**config)
        self.model.opt.disableflags = 128 ##IMP disables clamping of controls to the range in the XML, i.e., [-1, 1]
        self.action_space.low *= action_space_max
        self.action_space.high *= action_space_max
