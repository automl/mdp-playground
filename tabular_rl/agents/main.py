import yaml
import numpy as np
import random

from tabular_rl.utils import get_env
from tabular_rl.agents.Dyna_Q import dyna_q
from tabular_rl.agents.Q_learning import q_learning
from tabular_rl.agents.Double_Q_learning import double_q_learning
from tabular_rl.agents.Sarsa import sarsa, n_step_sarsa, sarsa_lambda

if __name__ == '__main__':
    with open("../config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    env_name = config["env_name"]
    agent_name = config["agent_name"]

    discount_factor = config["discount_factor"]
    alpha = config["alpha"]

    agent_config = config["agents"][agent_name]
    env_config = config["envs"][env_name]

    eval_eps = config["eval_eps"]
    seed = config["seed"]
    no_render = config["no_render"]

    episodes = agent_config["episodes"]
    try:
        timesteps_total = agent_config['timesteps_total']
    except KeyError:
        timesteps_total = None
    env_max_steps = agent_config["env_max_steps"]
    agent_eps_decay = agent_config["agent_eps_decay"]
    agent_eps = agent_config["agent_eps"]

    np.random.seed(seed)
    random.seed(seed)

    env = get_env(env_name=env_name, env_max_steps=env_max_steps)

    if not no_render:
        # Clear screen in ANSI terminal
        print('\033c')
        print('\x1bc')
    print(agent_name)

    # todo: make q_learning a class and pass config as a parameter
    # todo: perhaps write rl algorithm super class
    # todo: parameterize td_update with "max=True" for Q_learning vs SARSA (but perhaps first implement all algorithms then refactor, find commonalities etc.)
    # todo: implement Q function as a Variable
    # implement n-step SARSA, n-step q-learning?
    if agent_name == 'q_learning':
        train_data, test_data, num_steps = q_learning(env, episodes, timesteps_total=timesteps_total,
                                                      epsilon_decay=agent_eps_decay, epsilon=agent_eps,
                                                      discount_factor=discount_factor, alpha=alpha,
                                                      eval_every=eval_eps, render_eval=not no_render)
    elif agent_name == 'double_q_learning':
        train_data, test_data, num_steps = double_q_learning(env, episodes, timesteps_total=timesteps_total,
                                                             epsilon_decay=agent_eps_decay,
                                                             epsilon=agent_eps, discount_factor=discount_factor, alpha=alpha,
                                                             eval_every=eval_eps, render_eval=not no_render)
    elif agent_name == 'sarsa':
        train_data, test_data, num_steps = sarsa(env, episodes, timesteps_total=timesteps_total,
                                                 epsilon_decay=agent_eps_decay,
                                                 epsilon=agent_eps, discount_factor=discount_factor, alpha=alpha,
                                                 eval_every=eval_eps, render_eval=not no_render)
    elif agent_name == 'n_step_sarsa':
        n = agent_config['n']
        train_data, test_data, num_steps = n_step_sarsa(env, episodes, timesteps_total=timesteps_total,
                                                        epsilon_decay=agent_eps_decay,
                                                        epsilon=agent_eps, discount_factor=discount_factor, alpha=alpha,
                                                        eval_every=eval_eps, render_eval=not no_render, n=n)
    elif agent_name == 'sarsa_lambda':
        lambd = agent_config['lambd']
        parallel_eligibility_updates = agent_config['parallel_eligibility_updates']
        train_data, test_data, num_steps = sarsa_lambda(env, episodes, timesteps_total=timesteps_total,
                                                        epsilon_decay=agent_eps_decay,
                                                        epsilon=agent_eps, discount_factor=discount_factor, alpha=alpha,
                                                        eval_every=eval_eps, render_eval=not no_render, lambd=lambd,
                                                        parallel_eligibility_updates=parallel_eligibility_updates)
    elif agent_name == 'dyna_q':
        mem_size = agent_config['mem_size']
        model_samples = agent_config['model_samples']
        train_data, test_data, num_steps = dyna_q(env, episodes, timesteps_total=timesteps_total,
                                                  epsilon_decay=agent_eps_decay,
                                                  epsilon=agent_eps, discount_factor=discount_factor, alpha=alpha,
                                                  eval_every=eval_eps, render_eval=not no_render,
                                                  memory_size=mem_size, sample_n_steps_from_model=model_samples)

    else:
        raise NotImplementedError
