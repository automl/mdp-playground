import gym
import copy
import numpy as np
import sys
from gym.spaces import Box, Tuple
from gym.wrappers import AtariPreprocessing
from ray.rllib.env.atari_wrappers import wrap_deepmind, is_atari
from mdp_playground.envs.rl_toy_env import RLToyEnv
import warnings
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from PIL.Image import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM

# def get_gym_wrapper(base_class):


class GymEnvWrapper(gym.Env):
    """Wraps an OpenAI Gym environment to be able to modify its dimensions corresponding to MDP Playground. Please see [`example.py`](example.py) for some simple examples of how to use this class. The values for these dimensions are passed in a config dict as for mdp_playground.envs.RLToyEnv. The description for the supported dimensions below can be found in mdp_playground/envs/rl_toy_env.py.

    Currently supported dimensions:
        transition noise (for discrete environments)
        reward delay
        reward noise
        image_transforms

    The wrapper is pretty general and can be applied to any Gym Environment. The environment should be instantiated and passed as the 1st argument to the __init__ method of this class. If using this wrapper with Atari, additional keys may be added specifying either atari_preprocessing = True or wrap_deepmind_ray = True. These would use the AtariPreprocessing wrapper from OpenAI Gym or wrap_deepmind() wrapper from Ray Rllib.

    For AtariPreprocessing, additional key-value pairs specifying grayscale_obs and frame_skip may be provided. These have the same meaning as in AtariPreprocessing.

    """

    # Should not be a gym.Wrapper because 1) gym.Wrapper has member variables observation_space and action_space while here with irrelevant_features we would have multiple observation_spaces and this could cause conflict with code that assumes any subclass of gym.Wrapper should have these member variables.
    # However, it _should_ be at least a gym.Env
    # Does it need to be a subclass of base_class because some external code
    # may check if it's an AtariEnv, for instance, and do further stuff based
    # on that?

    def __init__(self, env, **config):
        self.config = copy.deepcopy(config)
        # self.env = config["env"]
        self.env = env

        seed_int = None
        if "seed" in config:
            seed_int = config["seed"]

        self.seed(seed_int)  # seed
        # IMP Move below code from here to seed()? Because if seed is called
        # during the run of an env, the expectation is that all obs., act. space,
        # etc. seeds are set? Only Atari in Gym seems to do something similar, the
        # others I saw there don't seem to set seed for obs., act. spaces.
        self.env.seed(
            seed_int
        )  # seed ###IMP Apparently Atari also has a seed. :/ Without this, for beam_rider(?), about 1 in 5 times I got reward of 88.0 and 44.0 the remaining times with the same action sequence!! With setting this seed, I got the same reward of 44.0 when I ran about 20 times.; ##TODO If this is really a wrapper, should it be modifying the seed of the env?
        obs_space_seed = self.np_random.randint(sys.maxsize)  # random
        act_space_seed = self.np_random.randint(sys.maxsize)  # random
        self.env.observation_space.seed(obs_space_seed)  # seed
        self.env.action_space.seed(act_space_seed)  # seed

        # if "dummy_eval" in config: #hack
        #     del config["dummy_eval"]
        if "delay" in config:
            self.delay = config["delay"]
            assert config["delay"] >= 0
            self.reward_buffer = [0.0] * (self.delay)
        else:
            self.delay = 0

        if "transition_noise" in config:
            self.transition_noise = config["transition_noise"]
            if config["state_space_type"] == "continuous":
                assert callable(self.transition_noise), (
                    "transition_noise must be a function when env is continuous, it was of type:"
                    + str(type(self.transition_noise))
                )
            else:
                assert self.transition_noise <= 1.0 and self.transition_noise >= 0.0, (
                    "transition_noise must be a value in [0.0, 1.0] when env is discrete, it was:"
                    + str(self.transition_noise)
                )
        else:
            if config["state_space_type"] == "discrete":
                self.transition_noise = 0.0
            else:
                self.transition_noise = lambda a: 0.0

        if "reward_noise" in config:
            if callable(config["reward_noise"]):
                self.reward_noise = config["reward_noise"]
            else:
                reward_noise_std = config["reward_noise"]
                self.reward_noise = lambda a: a.normal(0, reward_noise_std)
        else:
            self.reward_noise = None

        if "image_transforms" not in config:
            self.image_transforms = False
        else:
            assert config["state_space_type"] == "discrete", (
                "Image transforms are only applicable to discrete envs."
            )
            self.image_transforms = config["image_transforms"]
            if len(self.env.observation_space.shape) != 3:
                warnings.warn("The length of observation_space.shape ="\
                    + self.env.observation_space.shape + "It was expected"\
                    + "to be 3 for environments with image representations."
                )

            if "image_padding" in config:
                self.image_padding = config["image_padding"]
            else:
                self.image_padding = 20

            if "image_sh_quant" not in config:
                if "shift" in self.image_transforms:
                    warnings.warn(
                        "Setting image shift quantisation to the \
                    default of 1, since no config value was provided for it."
                    )
                    self.image_sh_quant = 1
                else:
                    self.image_sh_quant = None
            else:
                self.image_sh_quant = config["image_sh_quant"]

            if "image_ro_quant" not in config:
                if "rotate" in self.image_transforms:
                    warnings.warn(
                        "Setting image rotate quantisation to the \
                    default of 1, since no config value was provided for it."
                    )
                    self.image_ro_quant = 1
                else:
                    self.image_ro_quant = None
            else:
                self.image_ro_quant = config["image_ro_quant"]

            if "image_scale_range" not in config:
                if "scale" in self.image_transforms:
                    warnings.warn(
                        "Setting image scale range to the default \
                    of (0.5, 1.5), since no config value was provided for it."
                    )
                    self.image_scale_range = (0.5, 1.5)
                else:
                    self.image_scale_range = None
            else:
                self.image_scale_range = config["image_scale_range"]


        if (
            "wrap_deepmind_ray" in config and config["wrap_deepmind_ray"]
        ):  # hack ##TODO remove?
            self.env = wrap_deepmind(self.env, dim=42, framestack=True)
        elif "atari_preprocessing" in config and config["atari_preprocessing"]:
            self.frame_skip = 4  # default for AtariPreprocessing
            if "frame_skip" in config:
                self.frame_skip = config["frame_skip"]
            self.grayscale_obs = False
            if "grayscale_obs" in config:
                self.grayscale_obs = config["grayscale_obs"]

            if "image_width" in config:
                self.image_width = config["image_width"]
            else:
                self.image_width = 84  # Atari default

            # Use AtariPreprocessing with frame_skip
            # noop_max set to 1 because we want to keep the vanilla env as
            # deterministic as possible and setting it 0 was not allowed. ##TODO
            # noop_max=0 is poosible in new Gym version, so update Gym version.
            self.env = AtariPreprocessing(self.env, frame_skip=self.frame_skip, grayscale_obs=self.grayscale_obs, noop_max=1, screen_size=self.image_width)
            print("self.env.noop_max set to: ", self.env.noop_max)

        if "irrelevant_features" in config:
            # self.irrelevant_features =  config["irrelevant_features"]
            irr_toy_env_conf = config["irrelevant_features"]
            if "seed" not in irr_toy_env_conf:
                irr_toy_env_conf["seed"] = self.np_random.randint(sys.maxsize)  #random

            if config["state_space_type"] == "discrete":
                pass
            else:  # cont. env
                # This is a bit hacky because we need to define the state_space_dim
                # of the irrelevant toy env in the "base" config and not the nested irrelevant_features
                # dict inside the base config to be compatible with the config_processor of MDPP
                irr_toy_env_conf["state_space_dim"] = \
                    config["irr_state_space_dim"]  # #hack

            self.irr_toy_env = RLToyEnv(**irr_toy_env_conf)

            if config["state_space_type"] == "discrete":
                self.action_space = Tuple(
                    (self.env.action_space, self.irr_toy_env.action_space)
                )
                self.observation_space = Tuple(
                    (self.env.observation_space, self.irr_toy_env.observation_space)
                )  # TODO for image observations, concatenate to 1 obs. space here and in step() and reset()?
            else:  # cont. env # TODO Check the test case added for cont. irr features case and code for it in run_experiments.py.
                env_obs_low = self.env.observation_space.low
                env_obs_high = self.env.observation_space.high
                env_obs_dtype = env_obs_low.dtype
                env_obs_shape = env_obs_low.shape
                irr_env_obs_low = self.irr_toy_env.observation_space.low
                irr_env_obs_high = self.irr_toy_env.observation_space.high
                irr_env_obs_dtype = self.irr_toy_env.observation_space.low.dtype
                assert env_obs_dtype == irr_env_obs_dtype, (
                    "Datatypes of base env and irrelevant toy env should match. Were: "
                    + str(env_obs_dtype)
                    + ", "
                    + str(irr_env_obs_dtype)
                )
                ext_low = np.concatenate((env_obs_low, irr_env_obs_low))
                ext_high = np.concatenate((env_obs_high, irr_env_obs_high))
                self.observation_space = Box(
                    low=ext_low, high=ext_high, dtype=env_obs_dtype
                )

                env_act_low = self.env.action_space.low
                env_act_high = self.env.action_space.high
                env_act_dtype = env_act_low.dtype
                self.env_act_shape = env_act_low.shape
                assert (
                    len(self.env_act_shape) == 1
                ), "Length of shape of action space should be 1."
                irr_env_act_low = self.irr_toy_env.action_space.low
                irr_env_act_high = self.irr_toy_env.action_space.high
                irr_env_act_dtype = irr_env_act_low.dtype
                # assert env_obs_dtype == env_act_dtype, "Datatypes of obs. and act. of
                # base env should match. Were: " + str(env_obs_dtype) + ", " +
                # str(env_act_dtype) #TODO Apparently, observations are np.float64 and
                # actions np.float32 for Mujoco.
                ext_low = np.concatenate((env_act_low, irr_env_act_low))
                ext_high = np.concatenate((env_act_high, irr_env_act_high))
                self.action_space = Box(
                    low=ext_low, high=ext_high, dtype=env_act_dtype
                )  # TODO Use BoxExtended here and above?

            self.observation_space.seed(obs_space_seed)  # #seed
            self.action_space.seed(act_space_seed)  # #seed
        else:  # no irrelevant features

            self.action_space = self.env.action_space
            if self.image_transforms:
                env_obs_low = self.env.observation_space.low
                env_obs_high = self.env.observation_space.high
                env_obs_dtype = env_obs_low.dtype
                env_obs_shape = env_obs_low.shape
                ext_low_shape = (
                            env_obs_shape[0] + self.image_padding * 2,
                            env_obs_shape[1] + self.image_padding * 2,
                            env_obs_shape[2]
                    )
                # #hardcoded stuff next
                ext_low = np.zeros(shape=(ext_low_shape))
                ext_high = np.ones(shape=(ext_low_shape)) * 255
                self.observation_space = Box(
                    low=ext_low, high=ext_high, dtype=env_obs_dtype
                )

            else:  # no image transforms
                self.observation_space = self.env.observation_space

        self.total_episodes = 0

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
        self.total_transitions_episode += 1

        if (
            self.config["state_space_type"] == "discrete"
            and self.transition_noise > 0.0
        ):
            probs = (
                np.ones(shape=(self.env.action_space.n,))
                * self.transition_noise
                / (self.env.action_space.n - 1)
            )
            probs[action] = 1 - self.transition_noise
            old_action = action
            action = int(
                self.np_random.choice(self.env.action_space.n, size=1, p=probs)
            )  # random
            if old_action != action:
                # print("NOISE inserted", old_action, action)
                self.total_noisy_transitions_episode += 1
        else:  # cont. envs
            pass  # TODO
            # self.total_abs_noise_in_transition_episode += np.abs(noise_in_transition)

        if "irrelevant_features" in self.config:
            if self.config["state_space_type"] == "discrete":
                next_state, reward, done, info = self.env.step(action[0])
                next_state_irr, _, done_irr, _ = self.irr_toy_env.step(action[1])
                next_state = tuple([next_state, next_state_irr])
            else:
                # env_act_shape is the shape of the underlying env's action space and we
                # sub-select those dimensions from the total action space next and apply
                # to the underlying env:
                next_state, reward, done, info = self.env.step(
                    action[: self.env_act_shape[0]]
                )
                next_state_irr, _, done_irr, _ = self.irr_toy_env.step(
                    action[self.env_act_shape[0]:]
                )
                next_state = np.concatenate((next_state, next_state_irr))
        else:
            next_state, reward, done, info = self.env.step(action)

        if self.image_transforms:
            next_state = self.get_transformed_image(next_state)

        if done:
            # if episode is finished return the rewards that were delayed and not
            # handed out before ##TODO add test case for this
            reward = np.sum(self.reward_buffer)
        else:
            self.reward_buffer.append(reward)
            old_reward = reward
            reward = self.reward_buffer[0]
            # print("rewards:", self.reward_buffer, old_reward, reward)
            del self.reward_buffer[0]

        # random ###TODO Would be better to parameterise this in terms of state,
        # action and time_step as well. Would need to change implementation to
        # have a queue for the rewards achieved and then pick the reward that was
        # generated delay timesteps ago.
        noise_in_reward = (self.reward_noise(self.np_random) if self.reward_noise else 0)
        self.total_abs_noise_in_reward_episode += np.abs(noise_in_reward)
        self.total_reward_episode += reward
        reward += noise_in_reward

        return next_state, reward, done, info

    def reset(self):
        # on episode "end" stuff (to not be invoked when reset() called when
        # self.total_episodes = 0; end is in quotes because it may not be a true
        # episode end reached by reaching a terminal state, but reset() may have
        # been called in the middle of an episode):
        if not self.total_episodes == 0:
            print("Noise stats for previous episode num.: " +
                  str(self.total_episodes) +
                  " (total abs. noise in rewards, total abs. noise in transitions, total reward, total noisy transitions, total transitions): " +
                  str(self.total_abs_noise_in_reward_episode) +
                  " " +
                  str(self.total_abs_noise_in_transition_episode) +
                  " " +
                  str(self.total_reward_episode) +
                  " " +
                  str(self.total_noisy_transitions_episode) +
                  " " +
                  str(self.total_transitions_episode))

        # on episode start stuff:
        self.reward_buffer = [0.0] * (self.delay)

        self.total_episodes += 1

        self.total_abs_noise_in_reward_episode = 0
        self.total_abs_noise_in_transition_episode = (
            0  # only present in continuous spaces
        )
        self.total_noisy_transitions_episode = 0  # only present in discrete spaces
        self.total_reward_episode = 0
        self.total_transitions_episode = 0

        if "irrelevant_features" in self.config:
            if self.config["state_space_type"] == "discrete":
                reset_state = self.env.reset()
                reset_state_irr = self.irr_toy_env.reset()
                reset_state = tuple([reset_state, reset_state_irr])
            else:
                reset_state = self.env.reset()
                reset_state_irr = self.irr_toy_env.reset()
                reset_state = np.concatenate((reset_state, reset_state_irr))
        else:
            reset_state = self.env.reset()

        if self.image_transforms:
            reset_state = self.get_transformed_image(reset_state)

        return reset_state
        # return super(GymEnvWrapper, self).reset()

    def seed(self, seed=None):
        """Initialises the Numpy RNG for the environment by calling a utility for this in Gym.

        Parameters
        ----------
        seed : int
            seed to initialise the np_random instance held by the environment. Cannot use numpy.int64 or similar because Gym doesn't accept it.

        Returns
        -------
        int
            The seed returned by Gym
        """
        # If seed is None, you get a randomly generated seed from gym.utils...
        self.np_random, self.seed_ = gym.utils.seeding.np_random(seed)  # random
        print(
            "Env SEED set to: "
            + str(seed)
            + ". Returned seed from Gym: "
            + str(self.seed_)
        )

        return self.seed_

    def get_transformed_image(self, env_img):
        # ###TODO write tests

        height = self.env.observation_space.shape[0]
        width = self.env.observation_space.shape[1]
        image_padding = self.image_padding
        tot_width = width + image_padding * 2
        tot_height = height + image_padding * 2
        assert height == width, "Currently only square images are supported."
        if len(self.env.observation_space.shape) == 3:
            channels = self.env.observation_space.shape[2]
        elif len(self.env.observation_space.shape) == 2:
            channels = 1
        else:
            raise ValueError()


        sh_quant = self.image_sh_quant
        ro_quant = self.image_ro_quant
        scale_range = self.image_scale_range

        # Assumes that if 3rd tensor dim is 3 that image is RGB
        if channels == 3:  # #hardcoded
            image_ = Image.new(
                "RGB", (tot_width, tot_height)
            )  # Use RGB for textures / custom images
        else:
            image_ = Image.new(
                "L", (tot_width, tot_height)
            )  # Use L for black and white 8-bit pixels instead of RGB in case not using custom images
        draw = ImageDraw.Draw(image_)

        # Currently assumes image width = height ###TODO rename R variable
        R = width
        # Assume COM of env_img is in centre of whole image
        shift_w = int(tot_width / 2)
        shift_h = int(tot_height / 2)
        #
        # if "scale" in self.transforms:
        #     # max_R = 0.6 * min(self.width, self.height) / 2 # Not sure whether to make this depend on provided R as well
        #     # min_R = 0.1 * min(self.width, self.height) / 2 # /2 because it's R, 0.6
        #     # and 0.1 to allow some wiggle for shift below and not make too small
        #     max_R = scale_range[1] * R
        #     if int(max_R) > min(self.width, self.height) / 2:
        #         warnings.warn(
        #             "Maximum possible size of polygon might be too big for the given resolution. It's set to: "
        #             + str(max_R)
        #         )
        #     max_R = np.log(max_R)
        #     min_R = scale_range[0] * R
        #     if int(min_R) < 3:
        #         warnings.warn(
        #             "Minimum possible size of polygon might be too small and lead too much noise in image. It's set to: "
        #             + str(min_R)
        #         )
        #     min_R = np.log(min_R)
        #     log_sample = min_R + self.np_random.random() * (max_R - min_R)
        #     sample_ = np.exp(log_sample)
        #     R = int(sample_)
        #     # print("R", min_R, max_R)
        #
        if "shift" in self.image_transforms:
            max_shift_w = (tot_width - R) // 2
            max_shift_h = (tot_height - R) // 2
            add_shift_w = self.np_random.randint(-max_shift_w + 1, max_shift_w)
            add_shift_h = self.np_random.randint(-max_shift_h + 1, max_shift_h)
            # print("add_shift_w, add_shift_h", add_shift_w, add_shift_h)
            add_shift_w = int(add_shift_w / sh_quant) * sh_quant
            add_shift_h = int(add_shift_h / sh_quant) * sh_quant
            # print("add_shift_w, add_shift_h", add_shift_w, add_shift_h)
            shift_w += add_shift_w
            shift_h += add_shift_h


        img_arr_ = np.array(image_)
        sq_width = R
        if (
            sq_width % 2 == 1
        ):  # If sq_width is not even, it causes errors with the //2 below.
            sq_width += 1
        # tex_img = tex_img.resize((sq_width, sq_width))
        # tex_arr = np.array(tex_img)
        top_left = (
            shift_h - height // 2,
            shift_w - width // 2,
        )
        bottom_right = (
            shift_h + height // 2,
            shift_w + width // 2,
        )
        img_arr_[
            top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]
        ] = env_img

        image_ = Image.fromarray(img_arr_, "RGB")


        # Because numpy is row-major and Image is column major, need to transpose
        ret_arr = np.transpose(np.array(image_), axes=(1, 0, 2))
        # ret_arr = np.array(image_)
        return ret_arr

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

# from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
# from gym.envs.atari import AtariEnv
# ae = AtariEnv(**{'game': 'beam_rider', 'obs_type': 'image', 'frameskip': 1})
# aew = GymEnvWrapper(ae, **{'reward_noise': lambda a: a.normal(0, 0.1), 'transition_noise': 0.1, 'delay': 1, 'frame_skip': 4, "atari_preprocessing": True, "state_space_type": "discrete", 'seed': 0})
# ob = aew.reset()
# print(ob.shape)
# print(ob)
# total_reward = 0.0
# for i in range(200):
#     act = aew.action_space.sample()
#     next_state, reward, done, info = aew.step(act)
#     print(reward, done, act)
#     if reward > 10:
#         print("reward in step:", i, reward)
#     total_reward += reward
# print("total_reward:", total_reward)
# aew.reset()

# # AtariPreprocessing()
# # from ray.tune.registry import register_env
# # register_env("AtariEnvWrapper", lambda config: AtariEnvWrapper(**config))
