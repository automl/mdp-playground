import gym
import copy
from gym.wrappers import AtariPreprocessing

# def get_gym_wrapper(base_class):
    # '''Wraps an OpenAI Gym environment to be able to modify its meta-features corresponding to MDP Playground'''
    # Should not be a gym.Wrapper because 1) gym.Wrapper has member variables observation_space and action_space while here with irrelevant_dims we would have mutliple observation_spaces and this could cause conflict with code that assumes any subclass of gym.Wrapper should have these member variables.
    # However, it _should_ be at least a gym.Env
    # Does it need to be a subclass of base_class because some external code may check if it's an AtariEnv, for instance, and do further stuff based on that?

class GymEnvWrapper(gym.Env):
    def __init__(self, env, **config):
        self.config = copy.deepcopy(config)
        # self.env = config["env"]
        self.env = env

        # if "dummy_eval" in config: #hack
        #     del config["dummy_eval"]
        if "delay" in config:
            assert config["delay"] >= 0
            self.reward_buffer = [0] * (config["delay"])

        if "transition_noise" in config:
            self.transition_noise = config["transition_noise"]
            #next set seeds, assert for correct type of P, R noises

        if "reward_noise" in config:
            self.reward_noise =  config["reward_noise"]

        if config["atari_preprocessing"]:
            self.frame_skip = 4 # default for AtariPreprocessing
            if "frame_skip" in config:
                self.frame_skip = config["frame_skip"]
            if "grayscale_obs" in config:
                self.grayscale_obs = config["grayscale_obs"]
            self.env = AtariPreprocessing(self.env, frame_skip=self.frame_skip, grayscale_obs=self.grayscale_obs)
            # Use AtariPreprocessing with frame_skip

        if "irrelevant_dims" in config:
            self.irrelevant_dims =  config["irrelevant_dims"]
        else:
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space


        # if "action_loss_weight" in config: #hack
        #     del config["action_loss_weight"]
        # if "action_space_max" in config: #hack
        #     action_space_max = config["action_space_max"]
        #     del config["action_space_max"]
        # if "time_unit" in config: #hack
        #     time_unit = config["time_unit"]
        #     del config["time_unit"]
        # if "dummy_seed" in config: #hack
        #     del config["dummy_seed"]

        super(GymEnvWrapper, self).__init__()
        # if "action_space_max" in locals():
        #     print("Setting Mujoco self.action_space.low, self.action_space.high from:", self.action_space.low, self.action_space.high)
        #     self.action_space.low *= action_space_max
        #     self.action_space.high *= action_space_max
        #     print("to:", self.action_space.low, self.action_space.high)

            # if base_class == HalfCheetahEnv and action_space_max >= 4: #hack
            #     self.model.opt.timestep /= 2 # 0.005
            #     self.frame_skip *= 2
            #     print("Setting Mujoco timestep to", self.model.opt.timestep, "half of the usual to avoid instabilities. At the same time action repeat increased to twice its usual.")

        # if "time_unit" in locals(): #hack In HalfCheetah, this is needed because the reward function is dependent on the time_unit because it depends on velocity achieved which depends on amount of time torque was applied. In Pusher, Reacher, it is also needed because the reward is similar to the distance from current position to goal at _each_ step, which means if we calculate the reward multiple times in the same amount of "real" time, we'd need to average out the reward the more times we calculate the reward in the same amount of "real" time (i.e., when we have shorter acting timesteps). This is not the case with the toy enviroments because there the reward is amount of distance moved from current position to goal in the current timestep, so it's dependent on "real" time and not on acting timesteps.
            # self.frame_skip *= time_unit
            # self.frame_skip = int(self.frame_skip)
            # self._ctrl_cost_weight *= time_unit
            # self._forward_reward_weight *= time_unit
            # print("Setting Mujoco self.frame_skip, self._ctrl_cost_weight, self._forward_reward_weight to", self.frame_skip, self._ctrl_cost_weight, self._forward_reward_weight, "corresponding to time_unit in config.")

    def step(self, action):
        # next_state, reward, done, info = super(GymEnvWrapper, self).step(action)
        next_state, reward, done, info = self.env.step(action)
        self.reward_buffer.append(reward)
        delayed_reward = self.reward_buffer[0]
        del self.reward_buffer[0]
        return next_state, reward, done, info

    def reset(self):
        return self.env.reset()
        # return super(GymEnvWrapper, self).reset()

    # return GymEnvWrapper


# from mdp_playground.envs.gym_env_wrapper import get_gym_wrapper
# from gym.envs.atari import AtariEnv
# from gym.wrappers import AtariPreprocessing
# AtariPreprocessing()
# AtariEnvWrapper = get_gym_wrapper(AtariEnv)
# from ray.tune.registry import register_env
# register_env("AtariEnvWrapper", lambda config: AtariEnvWrapper(**config))
# aew = AtariEnvWrapper(**{'game': 'breakout', 'obs_type': 'image', 'frameskip': 4})
# ob = aew.reset()

# from gym.envs.atari import AtariEnv
# from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
# ae = AtariEnv(**{'game': 'breakout', 'obs_type': 'image', 'frameskip': 1})
# aew = GymEnvWrapper(ae, **{'reward_noise': 0.1, 'transition_noise': 0.1, 'delay': 1, 'frame_skip': 4, "atari_preprocessing": True})
# ob = aew.reset()
# print(ob.shape)
# print(ob)
# next_state, reward, done, info = aew.step(2)
# # AtariPreprocessing()
# # from ray.tune.registry import register_env
# # register_env("AtariEnvWrapper", lambda config: AtariEnvWrapper(**config))
