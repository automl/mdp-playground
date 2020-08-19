from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import copy

class HalfCheetahWrapperV3(HalfCheetahEnv):
    def __init__(self, **config):
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
            action_space_max = config["action_space_max"]
            del config["action_space_max"]
        if "time_unit" in config: #hack
            time_unit = config["time_unit"]
            del config["time_unit"]
        if "dummy_seed" in config: #hack
            del config["dummy_seed"]

        super(HalfCheetahWrapperV3, self).__init__(**config)
        self.model.opt.disableflags = 128 ##IMP disables clamping of controls to the range in the XML, i.e., [-1, 1]
        self.action_space.low *= action_space_max
        self.action_space.high *= action_space_max

        if "action_space_max" in locals() and action_space_max == 4: #hack
            self.model.opt.timestep /= 2 # 0.005
            self.frame_skip *= 2
            print("Setting Mujoco timestep to", self.model.opt.timestep, "half of the usual to avoid instabilities. At the same time action repeat increased to twice its usual.")

        if "time_unit" in locals(): #hack
            # self.model.opt.timestep /= 2 # 0.005
            self.frame_skip *= time_unit
            print("Setting Mujoco frame_skip to", self.frame_skip, "corresponding to time_unit in config.")
