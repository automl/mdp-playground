import numpy as np

from collections import defaultdict, deque
from tabular_rl.envs.Grid import GridCore
from tabular_rl.agents.rl_helpers import get_decay_schedule, make_epsilon_greedy_policy, td_update
from tabular_rl.agents.Q_learning import eval_policy


class SimpleMemory:
    """Simple Memory for Dyna-Q"""
    def __init__(self, size):
        self._size = size
        self._actions = deque([], maxlen=size)
        self._states = deque([], maxlen=size)
        self._added = 0

    def push(self, state, action):
        self._actions.append(action)
        self._states.append(state)
        self._added = min(self._added + 1, self._size)

    def sample(self):
        index = np.random.randint(self._added)
        return self._states[index], self._actions[index]


def dyna_q(
        environment: GridCore,
        num_episodes: int,
        timesteps_total: int = None,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True,
        sample_n_steps_from_model: int = 10,
        memory_size: int=1e4
):
    """
    Vanilla tabular Dyna-Q algorithm for deterministic environments
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
    :param sample_n_steps_from_model: Number of transitions to update with use of currentmodel.
    :param memory_size: Size of the replay buffer to train with the model.
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))
    Model = defaultdict(lambda: np.zeros((environment.action_space.n, 2)))
    memory = SimpleMemory(memory_size)

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
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        while True:  # roll out episode
            if init_timesteps_total is not None:  # Decay epsilon over timesteps
                epsilon = epsilon_schedule[min(num_performed_steps, timesteps_total - 1)]
                policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
            policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            num_performed_steps += 1
            if num_performed_steps >= timesteps_total:
                break
            memory.push(policy_state, policy_action)
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha, policy_done)
            Model[policy_state][policy_action][0] = policy_reward
            Model[policy_state][policy_action][1] = s_
            for model_sample_step in range(sample_n_steps_from_model):
                prior_taken_s, prior_taken_a = memory.sample()
                model_r, model_s_ = Model[prior_taken_s][prior_taken_a]
                Q[prior_taken_s][prior_taken_a] = td_update(Q, prior_taken_s, prior_taken_a, model_r, model_s_,
                                                            discount_factor, alpha, False)

            if init_timesteps_total is not None:
                if num_performed_steps % eval_every == 0:
                    test_rewards, test_lens, test_steps_list = eval_policy(
                        environment, Q, render_eval, test_rewards, test_lens, test_steps_list,
                        (num_performed_steps, timesteps_total, 'steps'))

            if policy_done:
                break
            policy_state = s_
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
