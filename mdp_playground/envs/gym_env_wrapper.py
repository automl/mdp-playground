import gym
import copy
import numpy as np
import sys
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

        seed_int = None
        if "seed" in config:
            seed_int = config["seed"]

        self.seed(seed_int)
        self.env.seed(seed_int) ###IMP Apparently Atari also has a seed. :/ Without this about 1 in 5 times I got reward of 88.0 and 44.0 the remaining times with the same action sequence!! With setting this seed, I got the same reward of 44.0 when I ran about 20 times.
        obs_space_seed = self.np_random.randint(sys.maxsize) #random
        act_space_seed = self.np_random.randint(sys.maxsize) #random
        self.env.observation_space.seed(obs_space_seed)
        self.env.action_space.seed(act_space_seed)

        # if "dummy_eval" in config: #hack
        #     del config["dummy_eval"]
        if "delay" in config:
            assert config["delay"] >= 0
            self.reward_buffer = [0.0] * (config["delay"])

        if "transition_noise" in config:
            self.transition_noise = config["transition_noise"]
            if config["state_space_type"] == "continuous":
                assert callable(self.transition_noise), "transition_noise must be a function when env is continuous, it was of type:" + str(type(self.transition_noise))
            else:
                assert self.transition_noise <= 1.0 and self.transition_noise >= 0.0, "transition_noise must be a value in [0.0, 1.0] when env is discrete, it was:" + str(self.transition_noise)
        else:
            self.transition_noise = 0.0
            #next set seeds, assert for correct type of P, R noises

        if "reward_noise" in config:
            self.reward_noise = config["reward_noise"]
        else:
            self.reward_noise = lambda a: 0.0

        if config["atari_preprocessing"]:
            self.frame_skip = 4 # default for AtariPreprocessing
            if "frame_skip" in config:
                self.frame_skip = config["frame_skip"]
            if "grayscale_obs" in config:
                self.grayscale_obs = config["grayscale_obs"]
            else:
                self.grayscale_obs = False
            # Use AtariPreprocessing with frame_skip
            self.env = AtariPreprocessing(self.env, frame_skip=self.frame_skip, grayscale_obs=self.grayscale_obs, noop_max=1)
            print("self.env.noop_max set to: ", self.env.noop_max)

        if "irrelevant_dims" in config:
            self.irrelevant_dims =  config["irrelevant_dims"]
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
        if self.config["state_space_type"] == "discrete" and self.transition_noise > 0.0:
            probs = np.ones(shape=(self.action_space.n,)) * self.transition_noise / (self.action_space.n - 1)
            probs[action] = 1 - self.transition_noise
            old_action = action
            action = int(self.np_random.choice(self.action_space.n, size=1, p=probs)) #random
            if old_action != action:
                # print("NOISE inserted", old_action, action)
                self.total_noisy_transitions_episode += 1
        else: # cont. envs
            pass ###TODO
            # self.total_abs_noise_in_transition_episode += np.abs(noise_in_transition)


        next_state, reward, done, info = self.env.step(action)
        self.reward_buffer.append(reward)
        old_reward = reward
        reward = self.reward_buffer[0]
        # print("rewards:", self.reward_buffer, old_reward, reward)
        del self.reward_buffer[0]

        noise_in_reward = self.reward_noise(self.np_random) #random ###TODO Would be better to parameterise this in terms of state, action and time_step as well. Would need to change implementation to have a queue for the rewards achieved and then pick the reward that was generated delay timesteps ago.
        self.total_abs_noise_in_reward_episode += np.abs(noise_in_reward)
        self.total_reward_episode += reward
        reward += noise_in_reward

        return next_state, reward, done, info

    def reset(self):
        # on episode "end" stuff (to not be invoked when reset() called when self.total_episodes = 0; end is in quotes because it may not be a true episode end reached by reaching a terminal state, but reset() may have been called in the middle of an episode):
        if not self.total_episodes == 0:
            print("Noise stats for previous episode num.: " + str(self.total_episodes) + " (total abs. noise in rewards, total abs. noise in transitions, total reward, total noisy transitions, total transitions): " + str(self.total_abs_noise_in_reward_episode) + " " + str(self.total_abs_noise_in_transition_episode) + " " + str(self.total_reward_episode) + " " + str(self.total_noisy_transitions_episode) + " " + str(self.total_transitions_episode))

        # on episode start stuff:
        self.total_episodes += 1

        self.total_abs_noise_in_reward_episode = 0
        self.total_abs_noise_in_transition_episode = 0 # only present in continuous spaces
        self.total_noisy_transitions_episode = 0 # only present in discrete spaces
        self.total_reward_episode = 0
        self.total_transitions_episode = 0

        return self.env.reset()
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
        self.np_random, self.seed_ = gym.utils.seeding.np_random(seed) #random
        print("Env SEED set to: " + str(seed) + ". Returned seed from Gym: " + str(self.seed_))
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
