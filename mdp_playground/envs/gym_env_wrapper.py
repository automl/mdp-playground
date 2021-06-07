import gym
import copy
import numpy as np
import sys
from gym.spaces import Box, Tuple
from gym.wrappers import AtariPreprocessing
from ray.rllib.env.atari_wrappers import wrap_deepmind, is_atari
from mdp_playground.envs.rl_toy_env import RLToyEnv

# def get_gym_wrapper(base_class):


class GymEnvWrapper(gym.Env):
    """Wraps an OpenAI Gym environment to be able to modify its dimensions corresponding to MDP Playground. The documentation for the supported dimensions below can be found in mdp_playground/envs/rl_toy_env.py.

    Currently supported dimensions:
        transition noise (discrete)
        reward delay
        reward noise

    Also supports wrapping with AtariPreprocessing from OpenAI Gym or wrap_deepmind from Ray Rllib.

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

            # Use AtariPreprocessing with frame_skip
            # noop_max set to 1 because we want to keep the vanilla env as
            # deterministic as possible and setting it 0 was not allowed. ##TODO
            # noop_max=0 is poosible in new Gym version, so update Gym version.
            self.env = AtariPreprocessing(self.env, frame_skip=self.frame_skip, grayscale_obs=self.grayscale_obs, noop_max=1, )
            print("self.env.noop_max set to: ", self.env.noop_max)

        if "irrelevant_features" in config:
            # self.irrelevant_features =  config["irrelevant_features"]
            irr_toy_env_conf = config["irrelevant_features"]
            if "seed" not in irr_toy_env_conf:
                irr_toy_env_conf["seed"] = self.np_random.randint(sys.maxsize)  # random

            self.irr_toy_env = RLToyEnv(**irr_toy_env_conf)

            if config["state_space_type"] == "discrete":
                self.action_space = Tuple(
                    (self.env.action_space, self.irr_toy_env.action_space)
                )
                self.observation_space = Tuple(
                    (self.env.observation_space, self.irr_toy_env.observation_space)
                )  # TODO for image observations, concatenate to 1 obs. space here and in step() and reset()?
            else:  # TODO Check the test case added for cont. irr features case and code for it in run_experiments.py.
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

            self.observation_space.seed(obs_space_seed)  # seed
            self.action_space.seed(act_space_seed)  # seed
        else:
            self.action_space = self.env.action_space
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
                next_state, reward, done, info = self.env.step(
                    action[: self.env_act_shape[0]]
                )
                next_state_irr, _, done_irr, _ = self.irr_toy_env.step(
                    action[self.env_act_shape[0]:]
                )
                next_state = np.concatenate((next_state, next_state_irr))
        else:
            next_state, reward, done, info = self.env.step(action)

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
