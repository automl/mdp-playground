# from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import copy

def get_mujoco_wrapper(base_class):
    '''Wraps a mujoco-py environment to be able to modify its XML attributes'''

    class MujocoEnvWrapperV3(base_class):
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

            super(MujocoEnvWrapperV3, self).__init__(**config)
            self.model.opt.disableflags = 128 ##IMP disables clamping of controls to the range in the XML, i.e., [-1, 1]
            if "action_space_max" in locals():
                print("Setting Mujoco self.action_space.low, self.action_space.high from:", self.action_space.low, self.action_space.high)
                self.action_space.low *= action_space_max
                self.action_space.high *= action_space_max
                print("to:", self.action_space.low, self.action_space.high)

                if base_class == HalfCheetahEnv and action_space_max >= 4: #hack
                    self.model.opt.timestep /= 2 # 0.005
                    self.frame_skip *= 2
                    print("Setting Mujoco timestep to", self.model.opt.timestep, "half of the usual to avoid instabilities. At the same time action repeat increased to twice its usual.")

            if "time_unit" in locals(): #hack In HalfCheetah, this is needed because the reward function is dependent on the time_unit because it depends on velocity achieved which depends on amount of time torque was applied. In Pusher, Reacher, it is also needed because the reward is similar to the distance from current position to goal at _each_ step, which means if we calculate the reward multiple times in the same amount of "real" time, we'd need to average out the reward the more times we calculate the reward in the same amount of "real" time (i.e., when we have shorter acting timesteps). This is not the case with the toy enviroments because there the reward is amount of distance moved from current position to goal in the current timestep, so it's dependent on "real" time and not on acting timesteps.
                self.frame_skip *= time_unit
                self.frame_skip = int(self.frame_skip)
                self._ctrl_cost_weight *= time_unit
                self._forward_reward_weight *= time_unit
                print("Setting Mujoco self.frame_skip, self._ctrl_cost_weight, self._forward_reward_weight to", self.frame_skip, self._ctrl_cost_weight, self._forward_reward_weight, "corresponding to time_unit in config.")

    return MujocoEnvWrapperV3
