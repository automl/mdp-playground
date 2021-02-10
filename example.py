'''We collect here some examples of basic usage for MDP Playground.

Calling this file as a script, invokes the following examples:
    one for basic discrete environments
    one for discrete environments with image representations
    one for continuous environments with reward function move to a target point
    one for continuous environments with reward function move along a line, where movement over the last seq_len steps, that is closer to being linear is rewarded more.

Many further examples can be found in test_mdp_playground.py.
For an example of how to use multiple meta-features together in discrete environments, please see the test case test_discrete_all_meta_features().
For an example of using irrelevant dimensions in discrete environments: test_discrete_multi_discrete_irrelevant_dimensions().
For examples of how to use the continuous environments, for reward function move_to_a_pt see test cases that have names beginning with: test_continuous_dynamics_target_point_...().
For an example with multiple use cases of how to use the continuous environments, for reward function move_along_a_line: test_continuous_dynamics_move_along_a_line().
'''

from mdp_playground.envs import RLToyEnv
import numpy as np

def discrete_environment_example():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "discrete"
    config["state_space_size"] = 8
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
    env = RLToyEnv(**config) # Calls env.reset() automatically. So, in general, there is no need to call it after this.

    # The environment maintains an augmented state which contains the underlying state used by the MDP to perform transitions and hand out rewards. We can fetch a dict containing the augmented state and current state like this:
    augmented_state_dict = env.get_augmented_state()
    state = augmented_state_dict['curr_state']

    print("Taking a step in the environment with a random action and printing the transition:")
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def discrete_environment_image_representations_example():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "discrete"
    config["state_space_size"] = 8
    config["image_representations"] = True
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

    # The environment maintains an augmented state which contains the underlying state used by the MDP to perform transitions and hand out rewards. We can fetch a dict containing the augmented state and current state like this:
    augmented_state_dict = env.get_augmented_state()
    state = augmented_state_dict['curr_state']

    print("Taking a step in the environment with a random action and printing the transition:")
    action = env.action_space.sample()
    next_state_image, reward, done, info = env.step(action)
    next_state = augmented_state_dict['curr_state'] # Underlying MDP state holds the current discrete state.
    print("sars', done =", state, action, reward, next_state, done)

    # Display the image observation associated with the next state
    from PIL import Image
    img1 = Image.fromarray(np.squeeze(next_state_image), 'L') # 'L' is used for black and white. squeeze() is used because the image is 3-D because frameworks like Ray expect the image to be 3-D.
    img1.show()

    env.close()


def continuous_environment_example_move_along_a_line():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "continuous"
    config["state_space_dim"] = 4
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
    state = env.reset()

    print("Taking a step in the environment with a random action and printing the transition:")
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def continuous_environment_example_move_to_a_point():
    config = {}
    config["seed"] = 0

    config["state_space_type"] = "continuous"
    config["state_space_dim"] = 2
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

    env.close()

if __name__ == "__main__":

    # Colour print
    set_ansi_escape = "\033[33;1m" # Yellow, bold
    reset_ansi_escape = "\033[0m"

    print(set_ansi_escape + "Running discrete environment\n" + reset_ansi_escape)
    discrete_environment_example()

    print(set_ansi_escape + "\nRunning discrete environment with image representations\n" + reset_ansi_escape)
    discrete_environment_image_representations_example()

    print(set_ansi_escape + "\nRunning continuous environment: move_to_a_point\n" + reset_ansi_escape)
    continuous_environment_example_move_to_a_point()

    print(set_ansi_escape + "\nRunning continuous environment: move_along_a_line\n" + reset_ansi_escape)
    continuous_environment_example_move_along_a_line()

    # Using gym.make()
    import mdp_playground
    import gym
    gym.make('RLToy-v0')
    gym.make('RLToy-v0', **{'state_space_size':8, 'action_space_size':8, 'state_space_type':'discrete', 'action_space_type':'discrete', 'terminal_state_density':0.25, 'completely_connected': True})
