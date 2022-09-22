import numpy as np
from collections import defaultdict
from typing import Optional


def get_decay_schedule(start_val: float, decay_start: int, num_steps: int, type_: str):
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_steps: Total number of steps to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_steps)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_steps - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_steps - decay_start), endpoint=True)])
    else:
        raise NotImplementedError


def make_epsilon_greedy_policy(Q: defaultdict, epsilon: float, nA: int, Q_b: Optional[defaultdict] = None) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    :param Q_b: optional second Q-function for double Q learning
    """

    if Q_b is None:
        def policy_fn(observation):
            policy = np.ones(nA) * epsilon / nA
            best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
                Q[observation] == Q[observation].max()
            ))
            policy[best_action] += (1 - epsilon)
            return policy
    else:
        def policy_fn(observation):
            policy = np.ones(nA) * epsilon / nA
            double_Q = np.add(Q[observation], Q_b[observation])
            best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
                double_Q == double_Q.max()
            ))
            policy[best_action] += (1 - epsilon)
            return policy

    return policy_fn


def td_update(q: defaultdict, state: int, action: int, reward: float, next_state: int, gamma: float, alpha: float,
              done: bool = False, q_b: Optional[defaultdict] = None, action_: Optional[int] = None, eligibility: Optional[float] = None):
    """ Simple TD update rule """

    if q_b is None:
        if action_ is None:
            # TD update
            best_next_action = np.random.choice(
                np.flatnonzero(q[next_state] == q[next_state].max()))  # greedy best next (with tie-breaking)
            td_target = reward + gamma * q[next_state][best_next_action]
        else:
            # SARSA update
            td_target = reward + gamma * q[next_state][action_]
    else:
        # Double Q-learning TD update
        best_next_action = np.random.choice(
            np.flatnonzero(q[next_state] == q[next_state].max()))  # greedy best next (with tie-breaking)
        td_target = reward + gamma * q_b[next_state][best_next_action]
    if not done:
        td_delta = td_target - q[state][action]
    else:
        td_delta = td_target  # - 0
    if eligibility:
        # Sarsa(lambda) update
        td_delta *= eligibility
    return q[state][action] + alpha * td_delta


def update_eligibility_trace(s, Q_state, E_state, td_delta, alpha, discount_factor, lambd):
    for a in range(Q_state.size):
        Q_state[a] += alpha * td_delta * E_state[a]
        E_state[a] *= discount_factor * lambd
    return Q_state, E_state, s