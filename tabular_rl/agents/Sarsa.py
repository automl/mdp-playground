import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count

from collections import defaultdict
from tabular_rl.envs.Grid import GridCore
from tabular_rl.agents.rl_helpers import get_decay_schedule, make_epsilon_greedy_policy, td_update, update_eligibility_trace


def eval_policy(environment, Q, render_eval, test_rewards, test_lens, test_steps_list, horizon, crit=()):
    test_steps = 0
    # if i_episode % eval_every == 0:
    policy_state = environment.reset()
    episode_length, cummulative_reward = 0, 0
    if render_eval:
        environment.render()
    while True:  # roll out episode
        policy_action = np.random.choice(np.flatnonzero(Q[policy_state] == Q[policy_state].max()))
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


def sarsa(
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
    Vanilla tabular SARSA algorithm
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
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

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
        else:  # Decay epsilon over timesteps
            epsilon = epsilon_schedule[min(num_performed_steps, timesteps_total - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        while True:  # roll out episode
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

            a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha, policy_done,
                                                       action_=a_)
            if init_timesteps_total is not None:
                if num_performed_steps % eval_every == 0:
                    test_rewards, test_lens, test_steps_list = eval_policy(
                        environment, Q, render_eval, test_rewards, test_lens, test_steps_list, horizon=horizon,
                        crit=(num_performed_steps, timesteps_total, 'steps'))

            if policy_done:
                break
            policy_state = s_
            policy_action = a_
            if init_timesteps_total is not None:
                # If we update epsilon every time-step we also have to update our policy every timestep
                policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        rewards.append(cummulative_reward)
        lens.append(episode_length)
        #train_steps_list.append(environment.total_steps)

        if init_timesteps_total is None:
            if i_episode % eval_every == 0:
                test_rewards, test_lens, test_steps_list = eval_policy(
                    environment, Q, render_eval, test_rewards, test_lens, test_steps_list, horizon=horizon,
                    crit=(i_episode, num_episodes, 'episodes'))

        if num_performed_steps >= timesteps_total:
            break

    if init_timesteps_total is None:
        print('Done %4d/%4d %s' % (i_episode, num_episodes, 'episodes'))
    else:
        print('Done %4d/%4d %s' % (num_performed_steps, timesteps_total, 'steps'))
    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list), timesteps_per_iteration_statistics


def n_step_sarsa(
        environment: GridCore,
        num_episodes: int,
        timesteps_total: int = None,
        n: int = 1,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True):
    """
    Vanilla tabular n-step SARSA algorithm
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param timesteps_total: if not set -> None. If set overwrites the num_episodes
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :param n: todo
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []
    num_performed_steps = 0
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
        else:  # Decay epsilon over timesteps
            epsilon = epsilon_schedule[min(num_performed_steps, timesteps_total - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))

        n_step_rewards, n_step_states, n_step_actions = [], [], []

        n_step_states.append(policy_state)
        n_step_actions.append(policy_action)

        T = np.iinfo(np.int32).max
        tau = 0

        while not (tau == T-1):  # roll out episode
            if episode_length < T:
                s_, policy_reward, policy_done, _ = environment.step(policy_action)
                num_performed_steps += 1
                if num_performed_steps >= timesteps_total:
                    break
                cummulative_reward += policy_reward

                n_step_rewards.append(policy_reward)
                n_step_states.append(s_)

                if policy_done:
                    T = episode_length + 1
                else:
                    a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
                    policy_action = a_
                    n_step_actions.append(policy_action)

            tau = episode_length - n + 1
            if tau >= 0:
                G = []
                for i in range(tau+1, min(tau+n, T) + 1):
                    G.append((discount_factor**(i-tau-1))*n_step_rewards[i - 1])
                G = np.sum(G)
                if tau + n < T:
                    G += (discount_factor**n) * Q[n_step_states[tau+n]][n_step_actions[tau+n]]

                q_state_to_be_updated = n_step_states[tau]
                q_action_to_be_updated = n_step_actions[tau]
                Q[q_state_to_be_updated][q_action_to_be_updated] += alpha * (G - Q[q_state_to_be_updated][q_action_to_be_updated])

            episode_length += 1
            if init_timesteps_total is not None:
                if num_performed_steps % eval_every == 0:
                    test_rewards, test_lens, test_steps_list = eval_policy(
                        environment, Q, render_eval, test_rewards, test_lens, test_steps_list,
                        (num_performed_steps, timesteps_total, 'steps'))
                # If we update epsilon every time-step we also have to update our policy every timestep
                policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)

        rewards.append(cummulative_reward)
        lens.append(episode_length)
        train_steps_list.append(environment.total_steps)

        if init_timesteps_total is None:
            if i_episode % eval_every == 0:
                test_rewards, test_lens, test_steps_list = eval_policy(
                    environment, Q, render_eval, test_rewards, test_lens, test_steps_list,
                    (i_episode, num_episodes, 'episodes'))
        if num_performed_steps >= timesteps_total:
            break
    if init_timesteps_total is None:
        print('Done %4d/%4d %s' % (i_episode, num_episodes, 'episodes'))
    else:
        print('Done %4d/%4d %s' % (num_performed_steps, timesteps_total, 'steps'))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)


def sarsa_lambda(
        environment: GridCore,
        num_episodes: int,
        timesteps_total: int = None,
        lambd: float = .5,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True,
        parallel_eligibility_updates=False):
    """
    Vanilla tabular SARSA (lambda) algorithm using eligibility traces according to Sutton's RL book (http://incompleteideas.net/book/first/ebook/node77.html)
    (beware: above reference contains a bug as the eligibility matrix needs to be reset in each episode)
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param timesteps_total: if not set -> None. If set overwrites the num_episodes
    :param lambd: specifies the lambda parameter
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'

    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []
    num_performed_steps = 0
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
        # initialize eligibility matrix
        E = defaultdict(lambda: np.zeros(environment.action_space.n))
        if init_timesteps_total is None:  # Decay epsilon over episodes
            epsilon = epsilon_schedule[min(i_episode, num_episodes - 1)]
        else:  # Decay epsilon over timesteps
            epsilon = epsilon_schedule[min(num_performed_steps, timesteps_total - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        while True:  # roll out episode
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            num_performed_steps += 1
            if num_performed_steps >= timesteps_total:
                break
            a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
            cummulative_reward += policy_reward
            episode_length += 1

            td_delta = (policy_reward + discount_factor * Q[s_][a_]) - Q[policy_state][policy_action]
            E[policy_state][policy_action] += 1

            if parallel_eligibility_updates:
                # parallel version is only faster for large state spaces
                pool = Pool(cpu_count())
                # starmap maintains initial order
                results = pool.starmap(update_eligibility_trace, [(s, Q[s], E[s], td_delta, alpha, discount_factor, lambd) for s in Q.keys()])
                pool.close()
                pool.join()
                for result in results:
                    Q_state, E_state, s = result
                    Q[s], E[s] = Q_state, E_state
            else:
                for s in Q.keys():
                    Q[s], E[s], s = update_eligibility_trace(s, Q_state=Q[s], E_state=E[s], td_delta=td_delta, alpha=alpha, discount_factor=discount_factor,
                                                               lambd=lambd)
            if init_timesteps_total is not None:
                if num_performed_steps % eval_every == 0:
                    test_rewards, test_lens, test_steps_list = eval_policy(
                        environment, Q, render_eval, test_rewards, test_lens, test_steps_list,
                        (num_performed_steps, timesteps_total, 'steps'))

            if policy_done:
                break
            policy_state = s_
            policy_action = a_
            if init_timesteps_total is not None:
                # If we update epsilon every time-step we also have to update our policy every timestep
                policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)

        rewards.append(cummulative_reward)
        lens.append(episode_length)
        #train_steps_list.append(environment.total_steps)

        if init_timesteps_total is None:
            if i_episode % eval_every == 0:
                test_rewards, test_lens, test_steps_list = eval_policy(
                    environment, Q, render_eval, test_rewards, test_lens, test_steps_list,
                    (i_episode, num_episodes, 'episodes'))
        if num_performed_steps >= timesteps_total:
            break
    if init_timesteps_total is None:
        print('Done %4d/%4d %s' % (i_episode, num_episodes, 'episodes'))
    else:
        print('Done %4d/%4d %s' % (num_performed_steps, timesteps_total, 'steps'))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)
