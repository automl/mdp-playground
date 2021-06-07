# from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.reacher import ReacherEnv
import copy


def get_mujoco_wrapper(base_class):
    """Wraps a mujoco-py environment to be able to modify its XML attributes and inject the dimensions from MDP Playground. The values for these dimensions are passed in a config dict as for mdp_playground.envs.RLToyEnv.

    Currently supported dimensions:
        time_unit
        action_space_max

    For both of these dimensions the scalar value passed in the dict is used to multiply the base environments values. For the Mujoco envs, the time_unit is achieved by multiplying the Gym Mujoco env's frame_skip and thus needs to be such that time_unit * frame_skip is an integer. The time_unit is NOT achieved by changing Mujoco's timestep because that would change the numerical integration done my Mujoco.

    """
    # TODO This makes a subclass and not a wrapper. Change name. Or make it a
    # wrapper by using composition? Some frameworks might need an instance of
    # this class to also be an instance of base_class?

    class MujocoEnvWrapper(base_class):
        def __init__(self, **config):  # Gets passed env_config from run_experiments.py
            self.config = copy.deepcopy(config)
            self.base_class = base_class
            if (
                "dummy_eval" in config
            ):  # hack We need to delete these from config because either Gym or Ray complains when extraneous configs present
                del config["dummy_eval"]
            if "transition_noise" in config:  # hack
                del config["transition_noise"]
            if "reward_noise" in config:  # hack
                del config["reward_noise"]
            if "action_loss_weight" in config:  # hack
                del config["action_loss_weight"]
            if "action_space_max" in config:  # hack
                action_space_max = config["action_space_max"]
                del config["action_space_max"]
            if "time_unit" in config:  # hack
                self.time_unit = config["time_unit"]
                del config["time_unit"]
            if "dummy_seed" in config:  # hack
                del config["dummy_seed"]

            if "MujocoEnv" not in config:
                config["MujocoEnv"] = {}

            # hack ###IMP This helps by sending only Mujoco specific config to Mujoco,
            # if anything else is sent by using **config, program crashes because each
            # 'MujocoEnv', e.g. HalfCheetahV3, has a static call signature for
            # __init__ ##TODO Thanks to this, now the above del statements can be
            # removed I think.
            super(MujocoEnvWrapper, self).__init__(**config["MujocoEnv"])
            self.model.opt.disableflags = 128  # IMP disables clamping of controls to the range in the XML, i.e., [-1, 1]
            if "action_space_max" in locals():
                print(
                    "Setting Mujoco self.action_space.low, self.action_space.high from:",
                    self.action_space.low,
                    self.action_space.high,
                )
                self.action_space.low *= action_space_max
                self.action_space.high *= action_space_max
                print("to:", self.action_space.low, self.action_space.high)

                if base_class == HalfCheetahEnv and action_space_max >= 4:  # hack
                    self.model.opt.timestep /= 2  # 0.005
                    self.frame_skip *= 2
                    print(
                        "Setting Mujoco timestep to",
                        self.model.opt.timestep,
                        "half of the usual to avoid instabilities. At the same time action repeat increased to twice its usual.",
                    )

            if (
                "time_unit" in self.config
            ):  # hack In HalfCheetah, this is needed because the reward function is dependent on the time_unit because it depends on velocity achieved which depends on amount of time torque was applied. In Pusher, Reacher, it is also needed because the reward is similar to the distance from current position to goal at _each_ step, which means if we calculate the reward multiple times in the same amount of "real" time, we'd need to average out the reward the more times we calculate the reward in the same amount of "real" time (i.e., when we have shorter acting timesteps). This is not the case with the toy enviroments because there the reward is amount of distance moved from current position to goal in the current timestep, so it's dependent on "real" time and not on acting timesteps.
                print("Original frame_skip for Mujoco Env:", self.frame_skip)
                self.frame_skip *= self.time_unit
                self.frame_skip = int(self.frame_skip)
                print(
                    "Setting Mujoco self.frame_skip to",
                    self.frame_skip,
                    "corresponding to time_unit in config.",
                )
                assert self.frame_skip > 0, (
                    "self.frame_skip was set to < 0. Please check your time_unit setting. It was: "
                    + str(self.time_unit)
                )

                if (
                    base_class == HalfCheetahEnv
                ):  # hack could include other similarly defined envs from Gym Mujoco
                    self._ctrl_cost_weight *= self.time_unit
                    self._forward_reward_weight *= self.time_unit
                    print(
                        "Setting Mujoco self._ctrl_cost_weight, self._forward_reward_weight to",
                        self._ctrl_cost_weight,
                        self._forward_reward_weight,
                        "corresponding to time_unit in config.",
                    )

        def step(self, action):  # hack
            obs, reward, done, info = super(MujocoEnvWrapper, self).step(action)
            if (
                self.base_class in [PusherEnv, ReacherEnv]
                and "time_unit" in self.config
            ):
                reward *= self.time_unit
            return obs, reward, done, info

    return MujocoEnvWrapper


# from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper #hack
#
# from gym.envs.mujoco.reacher import ReacherEnv
# ReacherWrapperV2 = get_mujoco_wrapper(ReacherEnv)
# config = {"time_unit": 0.2}
# rw2 = ReacherWrapperV2(**config)
# o = rw2.reset()
# rw2.seed(0)
# rw2.step(1)
