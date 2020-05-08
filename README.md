# MDP Playground
A python package to benchmark low-level dimensions of difficulties for RL agents.

## Installation
```
conda create -n <env_name> python=3.6
conda activate <env_name>
git clone git@github.com:RaghuSpaceRajan/gym-extensions-for-mdp-playground.git
cd gym-extensions-for-mdp-playground
pip install -e .
cd ..
git clone git@github.com:RaghuSpaceRajan/mdp-playground.git
cd mdp-playground
pip install -e .[extras]
```

## Getting started
Please see [`example.py`](example.py) for some simple examples of how to use the MDP environments in the package. For further details, please refer to the documentation in [`mdp_playground/envs/rl_toy_env.py`](mdp_playground/envs/rl_toy_env.py).

## Running experiments
You can run experiments (currently uses Ray RLLib) using:
```
python run_experiments.py -c <config_file> -e <exp_name> -n <config_num>
```
The exp_name is a prefix for the filenames of CSV files where stats for the experiments are recorded.
Each of the command line arguments has defaults. Please refer to the documentation inside [`run_experiments.py`] for further details. (Or run with the `-h` flag to bring up help.)

The config files for experiments from the [paper](https://arxiv.org/abs/1909.07750) are in the experiments directory.
The name of the file corresponding to an experiment is formed as: <ALGORITHM_NAME>_<META_FEATURE_NAMES>.py
The possible ALGORITHM_NAMEs are: dqn, rainbow, a3c and a3c_lstm
The possible META_FEATURE_NAMES are: seq_del (for delay and sequence length varied together), p_r_noises (for P and R noises varied together), sparsity (for varying reward density) and make_denser (for varying make_denser)
For example, for algorithm DQN when varying meta-features delay and sequence length, the corresponding experiment file is dqn_seq_del.py

## Plotting
To plot results from experiments, run jupyter-notebook and open plot_experiments.ipynb in Jupyter. There are instructions within each of the cells on how to generate and save plots.

## Troubleshooting
### Installation
#### Alternate method (by recreating the developers' conda env)
```
conda env create -n <env_name> -f py36_toy_rl.yml
conda activate <env_name>
git clone git@github.com:RaghuSpaceRajan/gym-extensions-for-mdp-playground.git
cd gym-extensions-for-mdp-playground
pip install -e .
git clone git@github.com:RaghuSpaceRajan/mdp-playground.git
cd mdp-playground
pip install -e .
```

## Citing
If you use MDP Playground in your work, please cite the following paper:

```bibtex
@article{rajan2019mdp,
    title={MDP Playground: Meta-Features in Reinforcement Learning},
    author={Raghu Rajan and Frank Hutter},
    year={2019},
    eprint={1909.07750},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
