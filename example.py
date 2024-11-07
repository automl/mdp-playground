"""We collect some examples of basic usage for MDP Playground in this script.
Example calls: 
python example.py --do_not_display_images --log_level INFO
python example.py --do_not_display_images --func_list discrete_environment_example    
Equivalent call with short flags:
python example.py -n -ll INFO
python example.py -n -f discrete_environment_example

Calling this file as a script, invokes the following examples:
    one for basic discrete environments
    one for discrete environments with image representations
    one for discrete environments with a diameter > 1 and image representations
    one for continuous environments with reward function move to a target point
    one for continuous environments with reward function move to a target point with irrelevant features and image representations
    one for continuous environments with reward function move along a line
    one for basic grid environments
    one for grid environments with reward_every_n_steps
    one for grid environments with image representations
    one for wrapping Atari env qbert
    one for wrapping Mujoco envs HalfCheetah, Pusher, Reacher
    one for wrapping MiniGrid env  # Currently commented out due to some errors
    one for wrapping ProcGen env  # Currently commented out due to some errors
    two examples at the end showing how to create toy envs using gym.make()

Many further examples can be found in test_mdp_playground.py.
For an example of how to use multiple dimensions together in discrete environments, please see the test case test_discrete_all_meta_features().
For an example of using irrelevant dimensions in discrete environments: test_discrete_multi_discrete_irrelevant_dimensions().
For examples of how to use the continuous environments, for reward function move_to_a_pt see test cases that have names beginning with: test_continuous_dynamics_target_point_...().
For an example with multiple use cases of how to use the continuous environments, for reward function move_along_a_line: test_continuous_dynamics_move_along_a_line().
"""

from mdp_playground.envs import RLToyEnv
import numpy as np

display_images = True

def display_image(obs, mode="RGB"):
    # Display the image observation associated with the next state
    from PIL import Image

    # Because numpy is row-major and Image is column major, need to transpose
    obs = obs.transpose(1, 0, 2)
    img1 = Image.fromarray(np.squeeze(obs), mode)  # squeeze() is
    # used because the image is 3-D because frameworks like Ray expect the image
    # to be 3-D.
    img1.show()
    return img1


def discrete_environment_example():
    """discrete environment example"""

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
    env = RLToyEnv(**config)  # Calls env.reset()[0] automatically. So, in general,
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
    next_state, reward, done, trunc, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def discrete_environment_image_representations_example():
    '''discrete environment with image representations example'''

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "discrete"
    config["action_space_size"] = 8
    config["image_representations"] = True
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
    next_state_image, reward, done, trunc, info = env.step(action)
    augmented_state_dict = env.get_augmented_state()
    next_state = augmented_state_dict["curr_state"]  # Underlying MDP state holds
    # the current discrete state.
    print("sars', done =", state, action, reward, next_state, done)

    env.close()

    if display_images:
        display_image(next_state_image, mode="L")


def discrete_environment_diameter_image_representations_example():
    '''discrete environment with diameter > 1 and image representations example'''

    config = {}
    config["seed"] = 3

    config["state_space_type"] = "discrete"
    config["action_space_size"] = 4
    config["image_representations"] = True
    config["delay"] = 1
    config["diameter"] = 2
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
    next_state_image, reward, done, trunc, info = env.step(action)
    augmented_state_dict = env.get_augmented_state()
    next_state = augmented_state_dict["curr_state"]  # Underlying MDP state holds
    # the current discrete state.
    print("sars', done =", state, action, reward, next_state, done)

    env.close()

    if display_images:
        display_image(next_state_image, mode="L")


def continuous_environment_example_move_to_a_point():
    '''continuous environment example: move to a point'''

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
    state = env.reset()[0].copy()

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
    action = env.action_space.sample()
    next_state, reward, done, trunc, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def continuous_environment_example_move_to_a_point_irrelevant_image():
    '''continuous environment example: move to a point with irrelevant features and image representations'''

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "continuous"
    config["state_space_dim"] = 4
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

    config["image_representations"] = True
    config["irrelevant_features"] = True
    config["relevant_indices"] = [0, 1]

    env = RLToyEnv(**config)
    state = env.reset()[0]
    augmented_state_dict = env.get_augmented_state()
    state = augmented_state_dict["curr_state"].copy()  # Underlying MDP state holds
    # the current continuous state.

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
    action = env.action_space.sample()
    next_state_image, reward, done, trunc, info = env.step(action)
    augmented_state_dict = env.get_augmented_state()
    next_state = augmented_state_dict["curr_state"].copy()  # Underlying MDP state holds
    # the current continuous state.
    print("sars', done =", state, action, reward, next_state, done)

    env.close()

    if display_images:
        img1 = display_image(next_state_image, mode="RGB")
        # img1.save("cont_env_irrelevant_image.pdf")


def continuous_environment_example_move_along_a_line():
    '''continuous environment example: move along a line'''

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
    state = env.reset()[0].copy()

    print(
        "Taking a step in the environment with a random action and printing "
        "the transition:"
    )
    action = env.action_space.sample()
    next_state, reward, done, trunc, info = env.step(action)
    print("sars', done =", state, action, reward, next_state, done)

    env.close()


def grid_environment_example():
    '''grid environment example: move towards a goal point'''

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
        next_obs, reward, done, trunc, info = env.step(action)
        next_state = env.get_augmented_state()["augmented_state"][-1]
        print("sars', done =", state, action, reward, next_state, done)
        state = next_state

    env.reset()[0]
    env.close()

def grid_environment_example_reward_every_n_steps():
    '''grid environment example: move towards a goal point but with sparser rewards using the reward_every_n_steps config'''

    config = {}
    config["seed"] = 0

    config["state_space_type"] = "grid"
    config["grid_shape"] = (8, 8)

    config["reward_function"] = "move_to_a_point"
    config["make_denser"] = True
    config["reward_every_n_steps"] = 3
    config["target_point"] = [5, 5]

    env = RLToyEnv(**config)

    state = env.get_augmented_state()["augmented_state"][-1]
    actions = [[0, 1], [-1, 0], [-1, 0], [1, 0], [0.5, -0.5], [1, 2], [1, 1], [0, 1]]

    for i in range(len(actions)):
        action = actions[i]
        next_obs, reward, done, trunc, info = env.step(action)
        next_state = env.get_augmented_state()["augmented_state"][-1]
        print("sars', done =", state, action, reward, next_state, done)
        state = next_state

    env.reset()[0]
    env.close()


def grid_environment_image_representations_example():
    '''grid environment example: move towards a goal point with image representations'''

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
        next_obs, reward, done, trunc, info = env.step(action)
        next_state = env.get_augmented_state()["augmented_state"][-1]
        print("sars', done =", state, action, reward, next_state, done)
        state = next_state

    env.reset()[0]
    env.close()

    if display_images:
        display_image(next_obs)


def atari_wrapper_example():
    '''wrapping Atari env qbert example'''

    config = {
        "seed": 0,
        "delay": 1,
        "transition_noise": 0.25,
        "reward_noise": lambda s, a, rng: rng.normal(0, 0.1),
        "state_space_type": "discrete",
    }

    from mdp_playground.envs import GymEnvWrapper
    import gymnasium as gym

    ae = gym.make("QbertNoFrameskip-v4")
    env = GymEnvWrapper(ae, **config)
    state = env.reset()[0]

    print(
        "Taking 10 steps in the environment with a random action and printing the transition:"
    )
    for i in range(10):
        action = env.action_space.sample()
        next_state, reward, done, trunc, info = env.step(action)
        print(
            "s.shape a r s'.shape, done =",
            state.shape,
            action,
            reward,
            next_state.shape,
            done,
        )
        state = next_state

    env.close()

    if display_images:
        display_image(next_state)


def mujoco_wrapper_examples():
    '''wrapping Mujoco envs HalfCheetah, Pusher, Reacher examples'''

    # For Mujoco envs, a few specific dimensions need to be changed by fiddling with 
    # attributes of the MujocoEnv class. This is achieved through a Mujoco
    # wrapper that subclasses the Mujoco env and modifies relevant properties.
    # Please see the documentation of mujoco_env_wrapper.py for more details. 
    # Below, we specify 2 dicts: one for the specific dimensions that are changed
    # using the Mujoco wrapper and the other for the general dimensions that are
    # changed using a GymEnvWrapper.

    # 1: Mujoco wrapper config:
    # The scalar values for the dimensions passed in this dict are used to
    # multiply the base environments' values. For these Mujoco envs, the
    # time_unit is achieved by multiplying the Gym Mujoco env's frame_skip and
    # thus will be the integer part of time_unit * frame_skip. (For HalfCheetah-v4
    # and Pusher-v4, frame_skip is 5; for Reacher-v4, it is 2.) The time_unit
    # is NOT achieved by changing Mujoco's timestep because that would change
    # the numerical integration done my Mujoco and thus the environment
    # dynamics.
    mujoco_wrap_config = {
        "action_space_max": 0.5,
        "time_unit": 0.5,
    }

    # 2: Gym wrapper config:
    gym_wrap_config = {
        "seed": 0,
        "state_space_type": "continuous",
        "transition_noise": 0.25,
    }


    # This makes a subclass and not a wrapper because some
    # frameworks might need an instance of this class to also be an instance
    # of the Mujoco base_class.
    try:
        from mdp_playground.envs import get_mujoco_wrapper

        # HalfCheetah example
        from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
        wrapped_mujoco_env = get_mujoco_wrapper(HalfCheetahEnv)

        env = wrapped_mujoco_env(**mujoco_wrap_config)

        from mdp_playground.envs import GymEnvWrapper
        import gymnasium as gym
        env = GymEnvWrapper(env, **gym_wrap_config)

        # From Gymnasium v26, the seed is set in the reset method.
        state = env.reset(seed=gym_wrap_config["seed"])[0]

        print(
            "Taking steps in the HalfCheetah environment with a random action and printing the transition:"
        )
        for i in range(3):
            action = env.action_space.sample()
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state

        env.close()

        # Pusher example
        from gymnasium.envs.mujoco.pusher_v4 import PusherEnv
        wrapped_mujoco_env = get_mujoco_wrapper(PusherEnv)

        env = wrapped_mujoco_env(**mujoco_wrap_config)

        from mdp_playground.envs import GymEnvWrapper
        import gymnasium as gym
        env = GymEnvWrapper(env, **gym_wrap_config)

        state = env.reset(seed=gym_wrap_config["seed"] + 1)[0]

        print(
            "Taking steps in the Pusher environment with a random action and printing the transition:"
        )
        for i in range(3):
            action = env.action_space.sample()
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state

        env.close()

        # Reacher example
        from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
        wrapped_mujoco_env = get_mujoco_wrapper(ReacherEnv)

        env = wrapped_mujoco_env(**mujoco_wrap_config)
        
        from mdp_playground.envs import GymEnvWrapper
        import gymnasium as gym
        env = GymEnvWrapper(env, **gym_wrap_config)

        state = env.reset(seed=gym_wrap_config["seed"] + 2)[0]

        print(
            "Taking steps in the Reacher environment with a random action and printing the transition:"
        )
        for i in range(3):
            action = env.action_space.sample()
            next_state, reward, done, trunc, info = env.step(action)
            print("sars', done =", state, action, reward, next_state, done)
            state = next_state

        env.close()

    except ImportError as e:
        print(
            "Exception:",
            type(e),
            e,
            "caught. You may need to install mujoco with pip. NOT running mujoco_wrapper_examples.",
        )
        return


def minigrid_wrapper_example():
    '''wrapping MiniGrid env example'''

    config = {
        "seed": 0,
        "delay": 1,
        "transition_noise": 0.25,
        "reward_noise": lambda s, a, rng: rng.normal(0, 0.1),
        "state_space_type": "discrete",
    }

    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    import gymnasium as gym

    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field

    env = GymEnvWrapper(env, **config)
    obs = env.reset()[0]  # This now produces an RGB tensor only

    print(
        "Taking a step in the environment with a random action and printing the transition:"
    )
    action = env.action_space.sample()
    next_obs, reward, done, trunc, info = env.step(action)
    print(
        "s.shape ar s'.shape, done =",
        obs.shape,
        action,
        reward,
        next_obs.shape,
        done,
    )

    env.close()

    if display_images:
        display_image(next_obs)


def procgen_wrapper_example():
    '''wrapping ProcGen env example'''

    config = {
        "seed": 0,
        "delay": 1,
        "transition_noise": 0.25,
        "reward_noise": lambda s, a, rng: rng.normal(0, 0.1),
        "state_space_type": "discrete",
    }

    from mdp_playground.envs.gym_env_wrapper import GymEnvWrapper
    import gymnasium as gym

    env = gym.make("procgen:procgen-coinrun-v0")
    env = GymEnvWrapper(env, **config)
    obs = env.reset()[0]

    print(
        "Taking a step in the environment with a random action and printing the transition:"
    )
    action = env.action_space.sample()
    next_obs, reward, done, trunc, info = env.step(action)
    print(
        "s.shape ar s'.shape, done =",
        obs.shape,
        action,
        reward,
        next_obs.shape,
        done,
    )

    env.close()

    if display_images:
        display_image(next_obs)


if __name__ == "__main__":

    # Use argparse to set display_images to False if you don't want to display images
    # and to set log level.
    import argparse
    parser = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--display_images", "-di", help="Display image observations (available for some examples)", action="store_true")
    parser.add_argument("--do_not_display_images", "-n", help="Do not display image observations (available for some examples)", action="store_false", dest="display_images")
    parser.add_argument("--log_level", "-ll", type=str, default="DEBUG", help="Set the log level")
    parser.add_argument("--func_list", "-f", type=str, nargs="+", help="Set the list of examples to run. Set it to the names of the functions corresponding to the examples inside this script.")
    parser.set_defaults(display_images=True)
    args = parser.parse_args()
    # print("Args:", args)
    display_images = args.display_images

    # Set up logging globally for the MDP Playground library:
    import logging
    logger = logging.getLogger("mdp_playground")
    logger.setLevel(args.log_level)
    if not logger.handlers:
        log_filename = "log_file.txt"
        log_file_handler = logging.FileHandler(log_filename)
        log_file_handler.setFormatter(logging.Formatter('%(message)s - %(levelname)s - %(name)s - %(asctime)s', datefmt='%m.%d.%Y %I:%M:%S %p'))
        logger.addHandler(log_file_handler)
        # Add a console handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        # Have less verbose logging to console:
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        logger.info("Begin logging to: %s", log_filename)


    # Colour print
    set_ansi_escape = "\033[33;1m"  # Yellow, bold
    reset_ansi_escape = "\033[0m"

    # Run the examples called in the function list:
    if args.func_list:
        for func_name in args.func_list:
            logger.info(set_ansi_escape + "Running " + globals()[func_name].__doc__ + reset_ansi_escape)
            globals()[func_name]()
        exit()

    # Else run all other examples except the ones disabled right now:

    # List all function names defined in the current script
    functions = [name for name, obj in globals().items() if callable(obj) and obj.__module__ == "__main__"]
    print("Available functions:", functions)

    # Disabled examples:
    functions_to_ignore = ["display_image", "minigrid_wrapper_example", "procgen_wrapper_example"]

    # Run all functions except the ones in functions_to_ignore:
    for func_name in functions:
        if func_name in functions_to_ignore:
            continue
        logger.info(set_ansi_escape + "Running " + globals()[func_name].__doc__ + reset_ansi_escape)
        globals()[func_name]()

    # Causes RuntimeError: dictionary changed size during iteration
    # global_vars = globals()
    # for func_name in global_vars:
    #     if callable(global_vars[func_name]):
    #         logger.info(func_name)

    # Running extra examples to show using gym.make():
    import mdp_playground
    import gymnasium as gym

    logger.info(set_ansi_escape + "Running 2 extra examples to show using gym.make()" + reset_ansi_escape)

    # The following are with seed=None:
    gym.make("RLToy-v0")

    env = gym.make(
        "RLToyFiniteHorizon-v0",
        **{
            "state_space_size": 8,
            "action_space_size": 8,
            "state_space_type": "discrete",
            "action_space_type": "discrete",
            "maximally_connected": True,
        }
    )
    env.reset()[0]
    for i in range(10):
        logger.info(env.step(env.action_space.sample()))
