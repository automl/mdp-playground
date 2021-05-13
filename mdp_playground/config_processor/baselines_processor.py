
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gym
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
import tensorflow as tf
import numpy as np
import stable_baselines as sb
from typing import Any, Dict, List, Optional, Union
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines import DQN, DDPG, SAC, A2C, TD3


# change config.agent_config,
# config.var_model_config
# config.var_agent_config to baselines framework
def agent_to_baselines(config):
    policy_kwargs = {}
    algorithm, agent_config, model_config, var_configs = config
    var_agent_configs, var_model_configs = {}, {}
    if("agent" in var_configs):
        var_agent_configs = var_configs["agent"]
    if("model" in var_configs):
        var_model_configs = var_configs["model"]


    valid_keys = ["gamma", "buffer_size", "batch_size", "learning_starts",
                  "timesteps_per_iteration"]  # common
    agent_to_model = []  # none
    # Check correct keys for each algorithm
    if algorithm == 'DQN':
        # change keys in dictionaries to something baselines understands
        valid_keys += ["exploration_fraction", "exploration_final_eps",
                       "exploration_initial_eps",
                       "train_freq", "double_q",
                       "prioritized_replay", "prioritized_replay_alpha",
                       "param_noise", "learning_rate",
                       "target_network_update_freq"]

        exchange_keys = [('noisy', 'param_noise'),
                         ('lr', 'learning_rate'),
                         ('train_batch_size', 'batch_size'),
                         ("rollout_fragment_length", "train_freq")]
        # Move dueling to policy parameters
        agent_to_model += [("dueling", "dueling")]
    elif algorithm == "DDPG":
        # memory_policy can be used to select PER
        # random_exploration, param_noise, action_noise parameters aditional
        valid_keys += ["critic_lr", "actor_lr", "tau","critic_l2_reg",\
                        "clip_norm", "nb_rollout_steps", "nb_train_steps"]
        valid_keys.remove("learning_starts") # For some reason it's the only one that does not have this implemented
        exchange_keys = [
                ('rollout_fragment_length','nb_rollout_steps'),
                ("l2_reg", "critic_l2_reg"),
                ("grad_norm_clipping", "clip_norm"),
                ('train_batch_size',   'batch_size')]
        # Because of how DDPG is implemented it will perform 100 rollout steps and then 50 train steps
        # This needs to be changed s.t. it does one rollout step and one training step
        agent_config["nb_rollout_steps"] = 1
        agent_config["nb_train_steps"] = 1
        agent_to_model+=[("actor_hiddens", "layers"),("critic_hiddens", "layers")]#cannot specify different nets
        # SPECIAL CASE, learning rates SHOULD NOT be none :c
        for key in ["critic_lr", "actor_lr"]:
            key_not_none = agent_config.get(key) 
            if (key in agent_config):
                if(not key_not_none): #key==none
                    agent_config.pop(key) #remove to get default value

    elif algorithm == "TD3":
        # memory_policy can be used to select PER; policy is always smoothened
        # random_exploration, param_noise, action_noise parameters aditional
        valid_keys += ["learning_rate", "policy_delay","tau","train_freq","gradient_steps",\
                       "target_policy_noise","target_noise_clip","action_noise"]
        exchange_keys = [
                ("target_noise", "target_policy_noise"),
                ("critic_lr", "learning_rate"),
                ("actor_lr", "learning_rate"),
                ('train_batch_size',   'batch_size')]
        agent_to_model += [("actor_hiddens", "layers"),( "critic_hiddens", "layers")]
        # SPECIAL CASE, learning rates SHOULD NOT be none :c
        for key in ["critic_lr, actor_lr"]:
            key_not_none = agent_config.get(key)
            if (key in agent_config):
                if(not key_not_none):
                    agent_config.pop(key)  # remove to get default value
        # Because of how TD3 is implemented it will perform 100 grad steps every 100 steps
        # This needs to be changed s.t. it does one rollout step and one training step
        agent_config["train_freq"] = 1
        agent_config["gradient_steps"] = 1
        # s.t. it clips actions between[-1,1]
        agent_config["action_noise"] = NormalActionNoise(mean=0, sigma=0)
    elif algorithm == "SAC":
        valid_keys += ["learning_rate", "tau", "ent_coef", "train_freq",
                       "target_update_interval", "clip_norm", "target_entropy"]
        exchange_keys = [
                ("rollout_fragment_length","train_freq"),
                ("entropy_learning_rate","learning_rate"),  # same learning rate for 
                ("critic_learning_rate", "learning_rate"),  # entropy 
                ("actor_learning_rate", "learning_rate"),  # actor and critic
                ("target_network_update_freq","target_update_interval"),
                ("grad_norm_clipping", "clip_norm"),
                ('train_batch_size',   'batch_size')]
        agent_to_model += [("fcnet_hiddens", "layers")]
        # -------- Add init entropy coef ------------#
        if("initial_alpha" in agent_config.keys()):
            agent_config["ent_coef"] = "auto_%.3f"%agent_config["initial_alpha"] #idk why but that's how baselines initializes this thing..
            agent_config.pop("initial_alpha")
        # var agent configs initial_alpha
        # idk why but that's how baselines initializes this thing..
        if(var_agent_configs):
            if("initial_alpha" in var_agent_configs):
                var_agent_configs["ent_coef"] = ["auto_%.3f"%coef for coef in var_agent_configs["initial_alpha"]]
                var_agent_configs.pop("initial_alpha")
        # ----------- special case ---------#
        # should be at least one
        if("target_network_update_freq" in agent_config):
            key_val = agent_config["target_network_update_freq"]
            agent_config["target_network_update_freq"] = 1 if key_val <= 0 else key_val 
        # -------- Take things out of optimization  ------------#
        # Special case: Optimization
        if("optimization" in agent_config):
            for k in agent_config["optimization"].keys():
                key_value = agent_config["optimization"][k]
                if(key_value):  # not none
                    agent_config[k] = key_value
        # -------- Take things out of Q-model and policy_model  ------------#
        # Can only specify one model for everything
        for move_key in ["Q_model", "policy_model"]:
            if(move_key in model_config):
                if("model" not in model_config):  # init
                    model_config["model"] = {}
                for k in model_config[move_key].keys():
                    model_config["model"][k] = model_config[move_key][k]
                model_config.pop(move_key)  # Remove from model_config
    else: #A2C
        algorithm = "A2C"
        valid_keys += ["learning_rate","vf_coef", "ent_coef","max_grad_norm"] #"lr_schedule" cannot be none
        exchange_keys = [
                ("vf_loss_coeff","vf_coef"),
                ("entropy_coeff","ent_coef"),
                ("lr", "learning_rate"),
                ("grad_clip", "max_grad_norm")]

    # -------- Change keys from Agent to Model dict when needed--------#
    for key_tuple in agent_to_model:
        old_key, new_key = key_tuple
        # -------- Agent config --------#
        key_exists = agent_config.get(old_key)
        if key_exists:  # If key exists and is not none
            policy_kwargs[new_key] = agent_config.pop(old_key)  # Move key to model config
        # #-------- Var agent config --------#
        key_exists = var_agent_configs.get(old_key)
        if key_exists:  # If key exists and is not none
            var_model_configs[new_key] = var_agent_configs.pop(old_key)  # Move key to model config

    if('fcnet_hiddens' in agent_config.keys()):
        policy_kwargs['layers'] = agent_config['fcnet_hiddens']
        agent_config.pop('fcnet_hiddens')

    # -------- Agent config --------#
    # change agent keys
    for key_tuple in exchange_keys:
        old_key, new_key = key_tuple
        # -------- Agent config --------#
        if(old_key in agent_config):
            agent_config[new_key] = agent_config.pop(old_key)
        # -------- Var agent config --------#
        if(old_key in var_agent_configs):
            var_agent_configs[new_key] = var_agent_configs.pop(old_key)

    # remove keys from dictionary which are not configurable by baselines
    for key in list(agent_config.keys()):
        if(key not in valid_keys):  # delete invalid keys
            agent_config.pop(key)

    # -------- Var agent config --------#
    # remove keys from dictionary which are not configurable by baselines
    for key in list(var_agent_configs.keys()):
        if(key not in valid_keys):#delete invalid keys
            var_agent_configs.pop(key) 

    # ----------- Model config ----------------#
    valid_keys = ["act_fun", "net_arch", "feature_extractor", "layers", "use_lstm","n_lstm"]
    exchange_keys = [("fcnet_activation", "act_fun"),
                     ("conv_activation", "act_fun"),
                     ("actor_hidden_activation", "act_fun"),
                     ("actor_hidden_activation", "act_fun"),
                     ("fcnet_hiddens", "layers"),
                     ("actor_hiddens", "layers"),
                     ("critic_hiddens","layers"),
                     ("conv_filters", "net_arch"),
                     ("lstm_cell_size","n_lstm")]

    # change keys
    feat_ext = 'mlp'
    for key_tuple in exchange_keys:
        old_key, new_key = key_tuple
        # -------- model config --------#
        if("model" in model_config):
            if(old_key in model_config["model"]):
                if(old_key == 'conv_filters'):
                    feat_ext = 'cnn'
                model_config["model"][new_key] = model_config["model"].pop(old_key)

        # -------- Var model config --------#
        if(old_key in var_model_configs):
            if(old_key == 'conv_filters'):
                feat_ext = 'cnn'
            var_model_configs[new_key] = var_model_configs.pop(old_key)

    # remove invalid keys
    if("model" in model_config):
        for key in list(model_config["model"].keys()):
            if(key not in valid_keys):
                model_config["model"].pop(key)

    # -------- Var model config - delete invalid keys -------#
    for key in list(var_model_configs.keys()):
        if(key not in valid_keys):
            var_model_configs.pop(key)

    # -------- set model config -------#
    policy_kwargs['feature_extraction'] = feat_ext
    agent_config['policy_kwargs'] = policy_kwargs
    # config.agent_config = agent_config
    # config.model_config = model_config
    # if bool(var_agent_configs): #variable configs
    #     config.var_configs["agent"] = var_agent_configs
    # if bool(var_model_configs):
    #     config.var_configs["model"] = var_model_configs
    return agent_config, model_config, var_agent_configs, var_model_configs


# change model_config to baselines framework/ decide MLP or CNN policy
def model_to_policy_kwargs(feat_ext, model_config):
    # -------------- Model configuration ------------#
    # # decide whether to use cnn or mlp, taken from ray code..
    # # Discrete/1D obs-spaces.
    # if isinstance(env.observation_space, gym.spaces.Discrete) or \
    #         len(env.observation_space.shape) <= 2:
    #     feat_ext = "mlp"
    # else:  # Default Conv2D net.
    #     feat_ext = "cnn"

    # Move model config(already on baselines framework) to policy_kwargs
    policy_kwargs, cnn_config = {}, {}
    for key in model_config["model"].keys():
        if(key == "act_fun"):
            policy_kwargs[key] = act_fnc_from_name(model_config["model"][key])
        else:
            policy_kwargs[key] = model_config["model"][key]

    if("use_lstm" in policy_kwargs.keys()):
        # use_lstm does not exist in baselines, here is used to define the policy
        use_lstm = policy_kwargs.pop("use_lstm")
    else:
        use_lstm = False

    if("n_lstm" in policy_kwargs.keys() and not use_lstm):
        policy_kwargs.pop("n_lstm")  # not valid if not lstm policy

    # Custom Feature extractor
    if(feat_ext == "cnn" and ("net_arch" in policy_kwargs)):
        cnn_config["act_fun"] = act_fnc_from_name(policy_kwargs.pop("act_fun"))
        cnn_config["net_arch"] = policy_kwargs.pop("net_arch")
        # Uncomment to pass through custom cnn
        policy_kwargs['cnn_extractor'] = vision_net
        policy_kwargs["kwargs"] = cnn_config

    policy_kwargs['feature_extraction'] = feat_ext 
    # # add policy arguments to agent configuration
    # if('lr_schedule'in agent_config.keys()): #schedule is part of model instead of agent in baselines
    #     agent_config['policy_kwargs']['lr_schedule'] = agent_config['lr_schedule']

    return policy_kwargs, use_lstm


def act_fnc_from_name(key):
    if(key == "tanh"):
        return tf.nn.tanh
    elif(key == "sigmoid"):
        return tf.nn.sigmoid
    elif(key == "leakyrelu"):
        return tf.nn.leaky_relu
    else: #default
        return tf.nn.relu


def vision_net(scaled_images, kwargs):
    if ("act_fun" in kwargs):
        activ_fn = kwargs['act_fun']
    else: #default
        activ_fn = tf.nn.relu

    output, i = scaled_images, 0
    for i, layer_def in enumerate(kwargs['net_arch'][:-1], start=1):
        n_filters, kernel, stride = layer_def
        scope = "c%d"%(i)
        output = activ_fn(conv(output, scope, n_filters = n_filters, filter_size = kernel[0], stride = stride, pad="SAME", init_scale=np.sqrt(2)))
    
    #last layer has valid padding
    n_filters, kernel, stride = kwargs['net_arch'][-1]
    output = activ_fn(conv(output, scope = "c%d"%(i+1), n_filters = n_filters, filter_size = kernel[0], stride = stride, pad="VALID", init_scale=np.sqrt(2)))
    output = conv_to_fc(output) #No linear layer
    #output = activ_fn(linear(output, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    return output


def init_agent(env, config_algorithm, agent_config_baselines, use_lstm):
    #Use feed forward policies and specify cnn feature extractor in configuration
    if config_algorithm == 'DQN':
        model = DQN(env = env, policy = sb.deepq.policies.FeedForwardPolicy, **agent_config_baselines, verbose=1)
    elif config_algorithm == 'DDPG':
        # action noise is none by default meaning it does not explore..
        # condiguration using ray defaults
        # https://docs.ray.io/en/latest/rllib-algorithms.html?highlight=ddpg#deep-deterministic-policy-gradients-ddpg-td3
        n_actions = env.action_space.shape[0]
        agent_config_baselines["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),\
                sigma=float(0.2) * np.ones(n_actions))
        #ddpg
        #can also use as agent parameter "normalize_observations = True " or "normalize_returns" = True
        model = DDPG(env = env, policy = sb.ddpg.policies.FeedForwardPolicy, **agent_config_baselines, verbose=1 )
    elif config_algorithm == "TD3":
        model = TD3(env = env, policy = sb.td3.policies.FeedForwardPolicy, **agent_config_baselines, verbose=1 )
    elif config_algorithm == "A3C" or config_algorithm == "A2C":
        policy = sb.common.policies.LstmPolicy if use_lstm else sb.common.policies.FeedForwardPolicy
        model = A2C(env = env, policy = policy ,**agent_config_baselines, verbose=1)
    else: #'SAC
        model = SAC(env = env, policy = sb.sac.policies.FeedForwardPolicy ,**agent_config_baselines, verbose=1)


class CustomCallback(sb.common.callbacks.BaseCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param deterministic: (bool) Whether the evaluation should use a stochastic or deterministic actions.
    :param verbose: (int)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_interval: int = 1,
        timesteps_per_iteration: int = 1000,
        deterministic: bool = True,
        verbose: int = 1,
        file_name: str = "./evaluation.csv",
        config_algorithm: str ="",
        var_configs = None
    ):
        super(CustomCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = timesteps_per_iteration * eval_interval
        self.deterministic = deterministic
        self.file_name = file_name
        self.timesteps_per_iteration = timesteps_per_iteration
        self.training_iteration = 1
        self.total_timesteps = 0
        self.config_algorithm = config_algorithm
        self.var_configs = var_configs
        self.last_episode_count = 0
        self.episodes_in_iter = 0
        self.best_eval_mean = 0
        # self.best_train_return = 0
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_obs=10.)

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env

    # evaluate policy when step % eval_freq == 0, return rewards and lengths list of n_eval_episodes elements
    def _on_step(self):
        # count steps
        self.total_timesteps+=1
        # training log
        # training iteration done, log train csv
        if(self.total_timesteps % self.timesteps_per_iteration == 0):
            self.write_train_result()
            self.training_iteration += 1

        # evaluation
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = sb.common.evaluation.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            # write evaluation csv
            self.write_eval_results(episode_rewards, episode_lengths)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            if(self.best_eval_mean < mean_reward):
                self.model.save("%s_best_eval" % (self.file_name))
                self.best_eval_mean = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        return True

    # def _on_rollout_end(self):
    #     env = self.training_env
    #     if isinstance(self.training_env, VecEnv):
    #         env = env.unwrapped.envs[0]  # only using one environment
    #     episode_rewards = env.get_episode_rewards()
    #     if(episode_rewards[-1] > self.best_train_return):
    #         self.model.save("%s_best_train" % (self.file_name))
    #         self.best_train_return = episode_rewards[-1]

    # Used in callback after every episode has ended during evaluation
    # replaces: def on_episode_end(info)
    def write_eval_results(self, episode_rewards, episode_lengths):
        eval_filename = self.file_name + '_eval.csv'
        fout = open(eval_filename, 'a')  # hardcoded
        for reward_this_episode, length_this_episode in zip(episode_rewards, episode_lengths):
            fout.write(str(reward_this_episode[0]) + ' ' + str(length_this_episode) + "\n")
        fout.close()

    # Write training stats to CSV file at end of every training iteration
    def write_train_result(self):
        env, training_iteration, total_timesteps = self.training_env, self.training_iteration, self.total_timesteps
        config_algorithm, var_configs = self.config_algorithm, self.var_configs

        if isinstance(env, VecEnv):
            env = env.unwrapped.envs[0]  # only using one environment

        #A2C can handle multiple envs..
        # if(self.config_algorithm == "A2C" or self.config_algorithm =="A3C"):
        #     env = env.envs[0]#take first env

        # Writes every iteration, would slow things down. #hack
        fout = open(self.file_name + '.csv', 'a') #hardcoded
        fout.write(str(training_iteration) + ' ' + config_algorithm + ' ')
        for config_type, config_dict in var_configs.items():
            for key in config_dict:
                if config_type == "env":
                    env_config = env.config
                    if key == 'reward_noise':
                        fout.write(str(env_config['reward_noise_std']) + ' ') #hack
                    elif key == 'transition_noise' and env_config["state_space_type"] == "continuous":
                        fout.write(str(env_config['transition_noise_std']) + ' ') #hack
                    else:
                        fout.write(str(env_config[key]).replace(' ', '') + ' ')
                elif config_type == "agent":
                    fout.write(str(getattr(self.model, key)).replace(' ', '') + ' ')
                elif config_type == "model":
                    if(key == "net_arch"):#this is kwargs as it is sent to visionet
                        fout.write(str(getattr(self.model, "policy_kwargs")["kwargs"][key]).replace(' ', '') + ' ')
                    else:
                        fout.write(str(getattr(self.model, "policy_kwargs")[key]).replace(' ', '') + ' ')

        # Write train statss
        episode_rewards = env.get_episode_rewards()
        episode_lengths = env.get_episode_lengths()
        # update episodes
        n_episodes = len(episode_rewards)
        self.episodes_in_iter = n_episodes - self.last_episode_count
        if(self.episodes_in_iter == 0):  # no episodes in iteration
            if(n_episodes) == 0:  # special case when there are no episodes so far
                episode_reward_mean = np.sum(env.rewards)  # cummulative reward so far
                episode_len_mean = env.get_total_steps()  # all steps so far
            else:
                # iteration end, no episode end
                # start = self.timesteps_per_iteration*training_iteration
                # episode_reward_mean = np.sum(env.rewards[start:])
                # episode_length_mean = self.timesteps_per_iteration
                episode_reward_mean = np.mean(episode_rewards)
                episode_length_mean = self.timesteps_per_iteration
        else:
            # episode stats are from all steps taken in env then we need to take mean over "iterations":
            # start , stop =  self.last_episode_count, self.last_episode_count + self.episodes_in_iter - 1
            episode_reward_mean = np.mean(episode_rewards[self.last_episode_count:])
            episode_length_mean = np.mean(episode_lengths[self.last_episode_count:])
        self.last_episode_count = n_episodes

        # timesteps_total always HAS to be the 1st written: analysis.py depends on it
        fout.write(str(total_timesteps) + ' ' +
                   str(episode_reward_mean) +
                   ' ' + str(episode_length_mean) + '\n')
        fout.close()

        # We did not manage to find an easy way to log evaluation stats for Ray without the following hack which demarcates the end of a training iteration in the evaluation stats file
        hack_filename_eval = self.file_name + '_eval.csv'
        fout = open(hack_filename_eval, 'a') #hardcoded
        fout.write('#HACK STRING EVAL' + "\n")
        fout.close()

        #info["result"]["callback_ok"] = True
        return True