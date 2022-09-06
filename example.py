"""We collect here some examples of basic usage for MDP Playground.

Calling this file as a script, invokes the following examples:
    one for basic discrete environments
    one for discrete environments with image representations
    one for continuous environments with reward function move to a target point
    one for basic grid environments
    one for grid environments with image representations
    one for wrapping Atari env qbert
    one for wrapping Mujoco env HalfCheetah
    two examples at the end showing how to create toy envs using gym.make()

Many further examples can be found in test_mdp_playground.py.
For an example of how to use multiple dimensions together in discrete environments, please see the test case test_discrete_all_meta_features().
For an example of using irrelevant dimensions in discrete environments: test_discrete_multi_discrete_irrelevant_dimensions().
For examples of how to use the continuous environments, for reward function move_to_a_pt see test cases that have names beginning with: test_continuous_dynamics_target_point_...().
For an example with multiple use cases of how to use the continuous environments, for reward function move_along_a_line: test_continuous_dynamics_move_along_a_line().
"""

from mdp_playground.envs import RLToyEnv
import numpy as np


def discrete_environment_example():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "discrete"
    config["action_space_size"] = 8
    config["delay"] = 1
    config["sequence_length"] = 3
    config["reward_scale"] = 2.5
    config["reward_shift"] = -1.75
    config["reward_noise"] = 0.5  # std dev of a Gaussian dist.
    config["transition_noise"] = 0.1
    config["reward_density"] = 0.25
    config["make_denser"] = False
    config["terminal_state_density"] = 0.25
    config["maximally_connected"] = True
    config["repeats_in_sequences"] = False

    config["generate_random_mdp"] = True
    env = RLToyEnv(**config)  # Calls env.reset() automatically. So, in general,
    # there is no need to call it after this.

    # The environment maintains an augmented state which contains the underlying
    # state used by the MDP to perform transitions and hand out rewards. We can
    # fetch a dict containing the augmented state and current state like this:
    augmented_state_dict = env.get_augmented_state()
    state = augmented_state_dict["curr_state"]

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def discrete_environment_image_representations_example():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "discrete"
    config["action_space_size"] = 8 # [8, 8]
    # config["irrelevant_features"] = True
    config["image_representations"] = True
    # config["image_transforms"] = "shift,scale,rotate,flip"
    config["delay"] = 1
    config["sequence_length"] = 3
    config["reward_scale"] = 2.5
    config["reward_shift"] = -1.75
    config["reward_noise"] = 0.5  # std dev of a Gaussian dist.
    config["transition_noise"] = 0.1
    config["reward_density"] = 0.25
    config["make_denser"] = False
    config["terminal_state_density"] = 0.25
    config["maximally_connected"] = True
    config["repeats_in_sequences"] = False

    config["generate_random_mdp"] = True
    env = RLToyEnv(**config)

    # The environment maintains an augmented state which contains the underlying
    # state used by the MDP to perform transitions and hand out rewards. We can
    # fetch a dict containing the augmented state and current state like this:
    augmented_state_dict = env.get_augmented_state()
    state = augmented_state_dict["curr_state"]

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
    action = env.action_space.sample()
    next_state_image, reward, done, info = env.step(action)
    next_state = augmented_state_dict["curr_state"]  # Underlying MDP state holds
    # the current discrete state.
    print("sars', done =", state, action, reward, next_state, done)

    # Display the image observation associated with the next state
    from PIL import Image

    # Because numpy is row-major and Image is column major, need to transpose
    next_state_image = next_state_image.transpose(1, 0, 2)
    img1 = Image.fromarray(np.squeeze(next_state_image), "L")  # 'L' is used for
    # black and white. squeeze() is used because the image is 3-D because
    # frameworks like Ray expect the image to be 3-D.
    img1.show()
    # img1.save("image_repr_irr_features_transforms.pdf")

    env.close()


def continuous_environment_example_move_along_a_line():

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "continuous"
    config["state_space_dim"] = 4
    config["transition_dynamics_order"] = 1
    config["inertia"] = 1  # 1 unit, e.g. kg for mass, or kg * m^2 for moment of
    # inertia.
    config["time_unit"] = 1  # Discretization of time domain and the time
    # duration over which action is applied

    config["delay"] = 0
    config["sequence_length"] = 10
    config["reward_scale"] = 1.0
    config["reward_noise"] = 0.1  # std dev of a Gaussian dist.
    config["transition_noise"] = 0.1  # std dev of a Gaussian dist.
    config["reward_function"] = "move_along_a_line"

    env = RLToyEnv(**config)
    state = env.reset()

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
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
    config["inertia"] = 1  # 1 unit, e.g. kg for mass, or kg * m^2 for moment of
    # inertia.
    config["time_unit"] = 1  # Discretization of time domain and the time
    # duration over which action is applied

    config["make_denser"] = True
    config["target_point"] = [0, 0]
    config["target_radius"] = 0.05
    config["state_space_max"] = 10
    config["action_space_max"] = 1
    config["action_loss_weight"] = 0.0

    config["reward_function"] = "move_to_a_point"

    env = RLToyEnv(**config)
    state = env.reset()

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def grid_environment_example():
    config = {}
    config["seed"] = 0

    config["state_space_type"] = "grid"
    config["grid_shape"] = (8, 8)

    config["reward_function"] = "move_to_a_point"
    config["make_denser"] = True
    config["target_point"] = [5, 5]

    env = RLToyEnv(**config)

    state = env.get_augmented_state()["augmented_state"][-1]
    actions = [[0, 1], [-1, 0], [-1, 0], [1, 0], [0.5, -0.5], [1, 2], [1, 1], [0, 1]]

    for i in range(len(actions)):
        action = actions[i]
        next_obs, reward, done, info = env.step(action)
        next_state = env.get_augmented_state()["augmented_state"][-1]
        print("sars', done =", state, action, reward, next_state, done)

    env.reset()
    env.close()


def grid_environment_image_representations_example():
    config = {}
    config["seed"] = 0

    config["state_space_type"] = "grid"
    config["grid_shape"] = (8, 8)

    config["reward_function"] = "move_to_a_point"
    config["make_denser"] = True
    config["target_point"] = [5, 5]

    config["image_representations"] = True
    config["terminal_states"] = [[5, 5], [2, 3], [2, 4], [3, 3], [3, 4]]
    env = RLToyEnv(**config)

    state = env.get_augmented_state()["augmented_state"][-1]
    actions = [[0, 1], [-1, 0], [-1, 0], [1, 0], [0.5, -0.5], [1, 2]]

    for i in range(len(actions)):
        action = actions[i]
        next_obs, reward, done, info = env.step(action)
        next_state = env.get_augmented_state()["augmented_state"][-1]
        print("sars', done =", state, action, reward, next_state, done)

    env.reset()
    env.close()

    # Display the image observation associated with the next state
    from PIL import Image

    # Because numpy is row-major and Image is column major, need to transpose
    next_obs = next_obs.transpose(1, 0, 2)
    img1 = Image.fromarray(np.squeeze(next_obs), "RGB")  # squeeze() is
    # used because the image is 3-D because frameworks like Ray expect the image
    # to be 3-D.
    img1.show()


def atari_wrapper_example():

    config = {
        "seed": 0,
        "delay": 1,
        "transition_noise": 0.25,
        "reward_noise": lambda a: a.normal(0, 0.1),
        "state_space_type": "discrete",
    }

    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    import gym

    ae = gym.make("QbertNoFrameskip-v4")
    env = GymEnvWrapper(ae, **config)
    state = env.reset()

    print(
        "Taking a step in the environment with a random action and printing the transition:"
    )
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(
        "s.shape ar s'.shape, done =",
        state.shape,
        action,
        reward,
        next_state.shape,
        done,
    )

    env.close()


def mujoco_wrapper_example():

    # The scalar values for the dimensions passed in this dict are used to
    # multiply the base environments' values. For these Mujoco envs, the
    # time_unit is achieved by multiplying the Gym Mujoco env's frame_skip and
    # thus will be the integer part of time_unit * frame_skip. The time_unit
    # is NOT achieved by changing Mujoco's timestep because that would change
    # the numerical integration done my Mujoco and thus the environment
    # dynamics.
    config = {
        "seed": 0,
        "action_space_max": 0.5,
        "time_unit": 0.5,
    }

    # This actually makes a subclass and not a wrapper. Because, some
    # frameworks might need an instance of this class to also be an instance
    # of the Mujoco base_class.
    from mdp_playground.envs.mujoco_env_wrapper import get_mujoco_wrapper
    from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

    wrapped_mujoco_env = get_mujoco_wrapper(HalfCheetahEnv)

    env = wrapped_mujoco_env(**config)
    state = env.reset()

    print(
        "Taking a step in the environment with a random action and printing the transition:"
    )
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


if __name__ == "__main__":

    # Colour print
    set_ansi_escape = "\033[33;1m"  # Yellow, bold
    reset_ansi_escape = "\033[0m"

    print(set_ansi_escape + "Running discrete environment\n" + reset_ansi_escape)
    discrete_environment_example()

    print(
        set_ansi_escape
        + "\nRunning discrete environment with image representations\n"
        + reset_ansi_escape
    )
    discrete_environment_image_representations_example()
    # import sys; sys.exit(0)

    print(
        set_ansi_escape
        + "\nRunning continuous environment: move_to_a_point\n"
        + reset_ansi_escape
    )
    continuous_environment_example_move_to_a_point()

    print(
        set_ansi_escape
        + "\nRunning grid environment: move_to_a_point\n"
        + reset_ansi_escape
    )
    grid_environment_example()

    print(
        set_ansi_escape + "\nRunning grid environment: move_to_a_point "
        "with image representations\n" + reset_ansi_escape
    )
    grid_environment_image_representations_example()

    print(set_ansi_escape + "\nRunning Atari wrapper example:\n" + reset_ansi_escape)
    atari_wrapper_example()

    print(set_ansi_escape + "\nRunning Mujoco wrapper example:\n" + reset_ansi_escape)
    mujoco_wrapper_example()

    # Using gym.make() example 1
    import mdp_playground
    import gym

    gym.make("RLToy-v0")

    env = gym.make(
        "RLToyFiniteHorizon-v0",
        **{
            "state_space_size": 8,
            "action_space_size": 8,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "terminal_state_density": 0.25,
            "maximally_connected": True,
        }
    )
    env.reset()
    for i in range(10):
        print(env.step(env.action_space.sample()))
