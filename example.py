'''We collect here some examples of basic usage for MDP Playground.

Calling this file as a script, invokes the following examples:
    one for discrete environments
    one for continuous environments

For an example of how to use multiple meta-features together in discrete environments, please see the test case test_discrete_all_meta_features().
For an example of using irrelevant dimensions in discrete environments: test_discrete_multi_discrete_irrelevant_dimensions().
For an example with multiple use cases of how to use the continuous environments, for reward function move_along_a_line: test_continuous_dynamics_move_along_a_line().
For examples of how to use the continuous environments, for reward function move_to_a_pt: test_continuous_dynamics_target_point_...().
'''

from mdp_playground.envs import RLToyEnv

def discrete_environment_example():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "discrete"
    config["action_space_type"] = "discrete"
    config["state_space_size"] = 8
    config["action_space_size"] = 8
    config["delay"] = 1
    config["sequence_length"] = 3
    config["reward_scale"] = 2.5
    config["reward_shift"] = -1.75
    config["reward_noise"] = lambda a: a.normal(0, 0.5)
    config["transition_noise"] = 0.1
    config["reward_density"] = 0.25
    config["make_denser"] = False
    config["terminal_state_density"] = 0.25
    config["completely_connected"] = True
    config["repeats_in_sequences"] = False

    config["generate_random_mdp"] = True
    env = RLToyEnv(**config)

    # The environment maintains an augmented state. We can fetch a dict containing the augmented state and current state like this:
    augmented_state_dict = env.get_augmented_state()
    state = augmented_state_dict['curr_state']

    print("Taking a step in the environment with a random action and printing the transition:")
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.reset()
    env.close()


def continuous_environment_example_move_along_a_line():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "continuous"
    config["action_space_type"] = "continuous"
    config["state_space_dim"] = 4
    config["action_space_dim"] = 4
    config["transition_dynamics_order"] = 1
    config["inertia"] = 1 # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
    config["time_unit"] = 1 # Discretization of time domain and the time duration over which action is applied

    config["delay"] = 0
    config["sequence_length"] = 10
    config["reward_scale"] = 1.0
    config["reward_noise"] = lambda a: a.normal(0, 0.1)
    config["transition_noise"] = lambda a: a.normal(0, 0.1)
    config["reward_function"] = "move_along_a_line"

    env = RLToyEnv(**config)
    state = env.get_augmented_state()['curr_state'].copy()

    print("Taking a step in the environment with a random action and printing the transition:")
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)
    state = next_state.copy()

    env.reset()
    env.close()


def continuous_environment_example_move_to_a_point():
    config = {}
    config["seed"] = 0

    config["state_space_type"] = "continuous"
    config["action_space_type"] = "continuous"
    config["state_space_dim"] = 2
    config["action_space_dim"] = 2
    config["transition_dynamics_order"] = 1
    config["inertia"] = 1 # 1 unit, e.g. kg for mass, or kg * m^2 for moment of inertia.
    config["time_unit"] = 1 # Discretization of time domain and the time duration over which action is applied

    config['make_denser'] = True
    config['target_point'] = [0, 0]
    config['target_radius'] = 0.05
    config["state_space_max"] = 10
    config["action_space_max"] = 1
    config["action_loss_weight"] = 0.0
    
    config["reward_function"] = "move_to_a_point"

    env = RLToyEnv(**config)
    state = env.reset()

    print("Taking a step in the environment with a random action and printing the transition:")
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)
    state = next_state.copy()

    env.reset()
    env.close()

if __name__ == "__main__":
    print("Running discrete environment")
    discrete_environment_example()
    print("\nRunning continuous environment: move_along_a_line")
    continuous_environment_example_move_along_a_line()
    print("\nRunning continuous environment: move_to_a_point")
    continuous_environment_example_move_to_a_point()

    # Using gym.make()
    import mdp_playground
    import gym
    gym.make('RLToy-v0')
    gym.make('RLToy-v0', **{'state_space_size':8, 'action_space_size':8, 'state_space_type':'discrete', 'action_space_type':'discrete', 'terminal_state_density':0.25, 'completely_connected': True})
