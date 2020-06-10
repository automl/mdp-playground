# MDP Playground
A python package to benchmark low-level dimensions of difficulties for RL agents.

## Getting started
There are 3 parts to the package:
1) **Environments**: The base Environment in [`mdp_playground/envs/rl_toy_env.py`](mdp_playground/envs/rl_toy_env.py) implements all the functionality, including discrete and continuous environments, and is parameterised by a `config` dict which contains all the information needed to instantiate the required MDP. Please see [`example.py`](example.py) for some simple examples of how to use the MDP environments in the package. For further details, please refer to the documentation in [`mdp_playground/envs/rl_toy_env.py`](mdp_playground/envs/rl_toy_env.py).

2) **Experiments**: Experiments are launched using [`run_experiments.py`](run_experiments.py). Config files for experiments are located inside the [`experiments`](experiments) directory. Please read the [instructions](#running-experiments) below for details.

3) **Analysis**: [`plot_experiments.ipynb`](plot_experiments.ipynb) contains code to plot the standard plots from the paper.

## Installation
**IMPORTANT**

We recommend using `conda` environments to manage virtual `Python` environments to run the experiments. Unfortunately, you will have to maintain 2 environments - 1 for **discrete** experiments and 1 for **continuous** experiments from the paper. As mentioned in Appendix H in the paper, this is because of issues with Ray, the library that we used for our baseline agents. Another reason to use a separate virtual environment is that our package `gym-extensions-for-mdp-playground` is currently a fork of `OpenAI Gym` and uses the same package name to avoid issues with other libraries such as `Ray`. We are waiting on a pull request for some of the clashes between our fork and `Gym` to resolve the problems.

Please follow the following commands to install for the discrete experiments:
```
conda create -n py36_toy_rl_disc python=3.6
conda activate py36_toy_rl_disc
cd gym-extensions-for-mdp-playground
pip install -e .
cd ..
cd mdp-playground
pip install -e .[extras_disc]
```

Please follow the following commands to install for the continuous experiments:
```
conda create -n py36_toy_rl_cont python=3.6
conda activate py36_toy_rl_cont
cd gym-extensions-for-mdp-playground
pip install -e .
cd ..
cd mdp-playground
pip install -e .[extras_cont]
wget 'https://ray-wheels.s3-us-west-2.amazonaws.com/master/8d0c1b5e068853bf748f72b1e60ec99d240932c6/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl'
pip install ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl[rllib,debug]
```

## Running experiments
For reproducing experiments from the main paper, please see [below](#running-experiments-from-main-paper).

For general instructions, please continue reading.
You can run experiments using:
```
python run_experiments.py -c <config_file> -e <exp_name> -n <config_num>
```
The `exp_name` is a prefix for the filenames of CSV files where stats for the experiments are recorded. The CSV stats files will be saved to the current directory.<br>
Each of the command line arguments has defaults. Please refer to the documentation inside [`run_experiments.py`](run_experiments.py) for further details on the command line arguments. (Or run it with the `-h` flag to bring up help.)

The config files for experiments from the [paper](https://arxiv.org/abs/1909.07750) are in the experiments directory.<br>
The name of the file corresponding to an experiment is formed as: `<algorithm_name>_<meta_feature_names>.py`<br>
The possible `algorithm_name`s are: `dqn`, `rainbow`, `a3c`, `a3c_lstm`, `ddpg`, `td3` and `sac`<br>
The possible `meta_feature_name`s are: `seq_del` (for **delay** and **sequence length** varied together), `p_r_noises` (for **P** and **R noises** varied together), `sparsity` (for varying **reward density**) and `make_denser` (for making **rewards denser** in environments with **sequences**)<br>
For example, for algorithm **DQN** when varying meta-features **delay** and **sequence length**, the corresponding experiment file is [`dqn_seq_del.py`](experiments/dqn_seq_del.py)

## Running experiments from the main paper
For completeness, we list here the commands for the experiments from the main paper:
```
# Discrete environments:
# We varied delay and sequence lengths together
conda activate py36_toy_rl_disc
python run_experiments.py -c experiments/dqn_seq_del.py -e dqn_seq_del
python run_experiments.py -c experiments/rainbow_seq_del.py -e rainbow_seq_del
python run_experiments.py -c experiments/a3c_seq_del.py -e a3c_seq_del
python run_experiments.py -c experiments/a3c_lstm_seq_del.py -e a3c_lstm_seq_del

# Representation learning:
python run_experiments.py -c experiments/dqn_image_representations.py -e dqn_image_representations
python run_experiments.py -c experiments/rainbow_image_representations.py -e rainbow_image_representations
python run_experiments.py -c experiments/a3c_image_representations.py -e a3c_image_representations
python run_experiments.py -c experiments/dqn_image_representations_sh_quant.py -e dqn_image_representations_sh_quant

# Continuous environments:
conda activate py36_toy_rl_cont
python run_experiments.py -c experiments/ddpg_move_to_a_point_target_radius.py -e ddpg_move_to_a_point_target_radius
python run_experiments.py -c experiments/ddpg_move_to_a_point_action_max.py -e ddpg_move_to_a_point_action_max
python run_experiments.py -c experiments/ddpg_move_to_a_point_time_unit.py -e ddpg_move_to_a_point_time_unit
python run_experiments.py -c experiments/ddpg_move_to_a_point_irr_dims.py -e ddpg_move_to_a_point_irr_dims
```
The CSV stats files will be saved to the current directory and can be analysed in [`plot_experiments.ipynb`](plot_experiments.ipynb).

## Plotting
To plot results from experiments, run `jupyter-notebook` and open [`plot_experiments.ipynb`](plot_experiments.ipynb) in Jupyter. There are instructions within each of the cells on how to generate and save plots.
