import numpy as np

from collections import defaultdict
from tabular_rl.envs.Grid import GridCore
from tabular_rl.agents.rl_helpers import get_decay_schedule, make_epsilon_greedy_policy, td_update


def eval_policy(environment, Q_a, Q_b, render_eval, test_rewards, test_lens, test_steps_list, horizon, crit=()):
    # evaluation with greedy policy
    test_steps = 0
    # if i_episode % eval_every == 0:
    policy_state = environment.reset()
    episode_length, cummulative_reward = 0, 0
    if render_eval:
        environment.render()
    while True:  # roll out episode
        double_Q = np.add(Q_a[policy_state], Q_b[policy_state])
        policy_action = np.random.choice(np.flatnonzero(double_Q == double_Q.max()))
        # environment.total_steps -= 1  # don't count evaluation steps
        s_, policy_reward, policy_done, _ = environment.step(policy_action)
        test_steps += 1
        if render_eval:
            environment.render()
        s_ = s_
        cummulative_reward += policy_reward
        episode_length += 1
        if policy_done or test_steps >= horizon:
            break
        policy_state = s_
    test_rewards.append(cummulative_reward)
    test_lens.append(episode_length)
    test_steps_list.append(test_steps)
    print('Done %4d/%4d %s' % crit)  # (i_episode, num_episodes, 'episodes'))
    return test_rewards, test_lens, test_steps_list


def double_q_learning(
        environment: GridCore,
        num_episodes: int,
        timesteps_total: int = None,
        horizon: int = 200,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True,
        rolling_window_mean_statistics: int = 10):
    """
    Double tabular Q-learning algorithm following
    Algorithm 1 from https://papers.nips.cc/paper/3964-double-q-learning.pdf
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param timesteps_total: if not set -> None. If set overwrites the num_episodes
    :param horizon: specifies the maximum number of steps within an episode
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :param rolling_window_mean_statistics: specifies the size n of the rolling window to compute the means over the last n rewards
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q_a = defaultdict(lambda: np.zeros(environment.action_space.n))
    Q_b = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    timesteps_per_iteration_statistics = []
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []
    num_performed_steps = 0
    num_performed_steps_per_episode = 0
    init_timesteps_total = timesteps_total

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts,
                                          num_episodes if timesteps_total is None else timesteps_total, epsilon_decay)

    # Determine if we evaluate based on episodes or total timesteps
    if timesteps_total is not None:
        num_episodes = np.iinfo(np.int32).max
    else:
        timesteps_total = np.iinfo(np.int32).max
    for i_episode in range(num_episodes + 1):
        # print('#' * 100)
        if init_timesteps_total is None:  # Decay epsilon over episodes
            epsilon = epsilon_schedule[min(i_episode, num_episodes - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q_a, epsilon, environment.action_space.n, Q_b=Q_b)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        while True:  # roll out episode
            if init_timesteps_total is not None:  # Decay epsilon over timesteps
                epsilon = epsilon_schedule[min(num_performed_steps, timesteps_total - 1)]
                policy = make_epsilon_greedy_policy(Q_a, epsilon, environment.action_space.n, Q_b=Q_b)
            policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            num_performed_steps += 1
            num_performed_steps_per_episode += 1

            if num_performed_steps % 1000 == 0:
                if (len(rewards) or len(lens)) == 0:
                    timesteps_per_iteration_statistics.append([num_performed_steps, 0., 0.])
                else:
                    timesteps_per_iteration_statistics.append([num_performed_steps, np.mean(rewards[-rolling_window_mean_statistics:]),
                                                               np.mean(lens[-rolling_window_mean_statistics:])])

            if num_performed_steps >= timesteps_total or num_performed_steps_per_episode >= horizon:
                num_performed_steps_per_episode = 0
                break

            cummulative_reward += policy_reward
            episode_length += 1

            if np.random.random() < 0.5:
                Q_a[policy_state][policy_action] = td_update(Q_a, policy_state, policy_action,
                                                             policy_reward, s_, discount_factor, alpha, policy_done,
                                                             Q_b)
            else:
                Q_b[policy_state][policy_action] = td_update(Q_b, policy_state, policy_action,
                                                             policy_reward, s_, discount_factor, alpha, policy_done,
                                                             Q_a)

            if init_timesteps_total is not None:
                if num_performed_steps % eval_every == 0:
                    test_rewards, test_lens, test_steps_list = eval_policy(
                        environment, Q_a, Q_b, render_eval, test_rewards, test_lens, test_steps_list, horizon=horizon,
                        crit=(num_performed_steps, timesteps_total, 'steps'))

            if policy_done:
                break
            policy_state = s_
        rewards.append(cummulative_reward)
        lens.append(episode_length)
        # train_steps_list.append(environment.total_steps)

        if init_timesteps_total is None:
            if i_episode % eval_every == 0:
                test_rewards, test_lens, test_steps_list = eval_policy(
                    environment, Q_a, Q_b, render_eval, test_rewards, test_lens, test_steps_list, horizon=horizon,
                    crit=(i_episode, num_episodes, 'episodes'))

        if num_performed_steps >= timesteps_total:
            break

    if init_timesteps_total is None:
        print('Done %4d/%4d %s' % (i_episode, num_episodes, 'episodes'))
    else:
        print('Done %4d/%4d %s' % (num_performed_steps, timesteps_total, 'steps'))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list), timesteps_per_iteration_statistics
