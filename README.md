<p align="center">

<a href="https://github.com/automl/mdp-playground/actions/workflows/gh-test.yml" target="_blank">
    <img src="https://github.com/automl/mdp-playground/actions/workflows/gh-test.yml/badge.svg" alt="Test">
</a>
<a href="https://github.com/automl/mdp-playground/actions/workflows/publish.yml" target="_blank">
    <img src="https://github.com/automl/mdp-playground/actions/workflows/publish.yml/badge.svg" alt="Publish">
</a>
<a href="https://codecov.io/gh/automl/mdp-playground" target="_blank">
    <img src="https://img.shields.io/codecov/c/github/automl/mdp-playground?color=%2334D058" alt="Coverage">
</a>
<a href="https://pypi.org/project/mdp-playground/" target="_blank">
    <img src="https://img.shields.io/pypi/v/mdp-playground?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/mdp-playground/" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/mdp-playground.svg" alt="Python Versions">
</a>
</p>


# MDP Playground
A python package to inject low-level dimensions of difficulties in RL environments. There are toy environments to design and debug RL agents. And complex environment wrappers for Atari and Mujoco to test robustness to these dimensions in complex environments.

## Getting started
There are 4 parts to the package:
1) **Toy Environments**: The base toy Environment in [`mdp_playground/envs/rl_toy_env.py`](mdp_playground/envs/rl_toy_env.py) implements the toy environment functionality, including discrete and continuous environments, and is parameterised by a `config` dict which contains all the information needed to instantiate the required MDP. Please see [`example.py`](example.py) for some simple examples of how to use the MDP environments in the package. For further details, please refer to the documentation in [`mdp_playground/envs/rl_toy_env.py`](mdp_playground/envs/rl_toy_env.py).

2) **Complex Environment Wrappers**: Similar to the toy environment, this is parameterised by a `config` dict which contains all the information needed to inject the dimensions into Atari or Mujoco environments. Please see [`example.py`](example.py) for some simple examples of how to use these. The Atari wrapper is in [`mdp_playground/envs/gym_env_wrapper.py`](mdp_playground/envs/gym_env_wrapper.py) and the Mujoco wrapper is in [`mdp_playground/envs/mujoco_env_wrapper.py`](mdp_playground/envs/mujoco_env_wrapper.py).

3) **Experiments**: Experiments are launched using [`run_experiments.py`](run_experiments.py). Config files for experiments are located inside the [`experiments`](experiments) directory. Please read the [instructions](#running-experiments) below for details on how to launch experiments.

4) **Analysis**: [`plot_experiments.ipynb`](plot_experiments.ipynb) contains code to plot the standard plots from the paper.


## Running experiments from the main paper
For reproducing experiments from the main paper, please continue reading.

For general instructions, please see [here](#installation).

### Installation for running experiments from the main paper
We recommend using `conda` environments to manage virtual `Python` environments to run the experiments. Unfortunately, you will have to maintain 2 environments - 1 for the "older" **discrete toy** experiments and 1 for the "newer" **continuous and complex** experiments from the paper. As mentioned in Appendix section **Tuned Hyperparameters** in the paper, this is because of issues with Ray, the library that we used for our baseline agents.

Please follow the following commands to install for the discrete toy experiments:
```bash
conda create -n py36_toy_rl_disc_toy python=3.6
conda activate py36_toy_rl_disc_toy
cd mdp-playground
pip install -r requirements.txt
pip install -e .[extras_disc]
```

Please follow the following commands to install for the continuous and complex experiments. **IMPORTANT**: In case, you do not have MuJoCo, please ignore any mujoco-py related installation errors below:
```bash
conda create -n py36_toy_rl_cont_comp python=3.6
conda activate py36_toy_rl_cont_comp
cd mdp-playground
pip install -r requirements.txt
pip install -e .[extras_cont]
wget 'https://ray-wheels.s3-us-west-2.amazonaws.com/master/8d0c1b5e068853bf748f72b1e60ec99d240932c6/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl'
pip install ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl[rllib,debug]
```

We list here the commands for the experiments from the main paper:
```bash
# Discrete toy environments:
# Image representation experiments:
conda activate py36_toy_rl_disc_toy
python run_experiments.py -c experiments/dqn_image_representations.py -e dqn_image_representations
python run_experiments.py -c experiments/rainbow_image_representations.py -e rainbow_image_representations
python run_experiments.py -c experiments/a3c_image_representations.py -e a3c_image_representations
python run_experiments.py -c experiments/dqn_image_representations_sh_quant.py -e dqn_image_representations_sh_quant

# Continuous toy environments:
conda activate py36_toy_rl_cont_comp
python run_experiments.py -c experiments/ddpg_move_to_a_point_time_unit.py -e ddpg_move_to_a_point_time_unit
python run_experiments.py -c experiments/ddpg_move_to_a_point_irr_dims.py -e ddpg_move_to_a_point_irr_dims
# Varying the action range and time unit together for transition_dynamics_order = 2
python run_experiments.py -c experiments/ddpg_move_to_a_point_p_order_2.py -e ddpg_move_to_a_point_p_order_2

# Complex environments:
# The commands below run all configs serially.
# In case, you want to parallelise on a cluster, please provide the CLI argument -n <config_number> at the end of the given commands. Please refer to the documentation for run_experiments.py for this.
conda activate py36_toy_rl_cont_comp
python run_experiments.py -c experiments/dqn_qbert_del.py -e dqn_qbert_del
python run_experiments.py -c experiments/ddpg_halfcheetah_time_unit.py -e ddpg_halfcheetah_time_unit

# For the spider plots, experiments for all the agents and dimensions will need to be run from the experiments directory, i.e., for discrete: dqn_p_r_noises.py, a3c_p_r_noises, ..., dqn_seq_del, ..., dqn_sparsity, ..., dqn_image_representations, ...
# for continuous:, ddpg_move_to_a_point_p_noise, td3_move_to_a_point_p_noise, ..., ddpg_move_to_a_point_r_noise, ..., ddpg_move_to_a_point_irr_dims, ..., ddpg_move_to_a_point_action_loss_weight, ..., ddpg_move_to_a_point_action_max, ..., ddpg_move_to_a_point_target_radius, ..., ddpg_move_to_a_point_time_unit
# and then follow the instructions in plot_experiments.ipynb

# For the bsuite debugging experiment, please run the bsuite sonnet dqn agent on our toy environment while varying reward density. Commit https://github.com/deepmind/bsuite/commit/5116216b62ce0005100a6036fb5397e358652530 should work fine.
```

The CSV stats files will be saved to the current directory and can be analysed in [`plot_experiments.ipynb`](plot_experiments.ipynb).


## Installation
For reproducing experiments from the main paper, please see [here](#running-experiments-from-the-main-paper).

### Production use
We recommend using `conda` to manage environments. After setup of the environment, you can install MDP Playground in two ways:
#### Manual
To install MDP Playground manually, clone the repository and run:
```bash
pip install -e .[extras]
```
This might be the preferred way if you want easy access to the included experiments.

#### From PyPI
MDP Playground is also on PyPI. Just run:
```bash
pip install mdp_playground[extras]
```


## Running experiments
You can run experiments using:
```
run-mdpp-experiments -c <config_file> -e <exp_name> -n <config_num>
```
The `exp_name` is a prefix for the filenames of CSV files where stats for the experiments are recorded. The CSV stats files will be saved to the current directory.<br>
Each of the command line arguments has defaults. Please refer to the documentation inside [`run_experiments.py`](run_experiments.py) for further details on the command line arguments. (Or run it with the `-h` flag to bring up help.)

The config files for experiments from the [paper](https://arxiv.org/abs/1909.07750) are in the experiments directory.<br>
The name of the file corresponding to an experiment is formed as: `<algorithm_name>_<dimension_names>.py`<br>
Some sample `algorithm_name`s are: `dqn`, `rainbow`, `a3c`, `a3c_lstm`, `ddpg`, `td3` and `sac`<br>
Some sample `dimension_name`s are: `seq_del` (for **delay** and **sequence length** varied together), `p_r_noises` (for **P** and **R noises** varied together),
`target_radius` (for varying **target radius**) and `time_unit` (for varying **time unit**)<br>
For example, for algorithm **DQN** when varying dimensions **delay** and **sequence length**, the corresponding experiment file is [`dqn_seq_del.py`](experiments/dqn_seq_del.py)

The CSV stats files will be saved to the current directory and can be analysed in [`plot_experiments.ipynb`](plot_experiments.ipynb).

## Plotting
To plot results from experiments, run `jupyter-notebook` and open [`plot_experiments.ipynb`](plot_experiments.ipynb) in Jupyter. There are instructions within each of the cells on how to generate and save plots.

## Citing
If you use MDP Playground in your work, please cite the following paper:

```bibtex
@article{rajan2020mdp,
      title={MDP Playground: Controlling Dimensions of Hardness in Reinforcement Learning},
      author={Raghu Rajan and Jessica Lizeth Borja Diaz and Suresh Guttikonda and Fabio Ferreira and André Biedenkapp and Frank Hutter},
      year={2020},
      eprint={1909.07750},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
