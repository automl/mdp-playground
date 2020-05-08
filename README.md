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
Please see [`example.py`](example.py) for some simple examples of how to use the package. For further details, please refer to the documentation in [`mdp_playground/envs/rl_toy_env.py`](mdp_playground/envs/rl_toy_env.py).

## Running experiments
You can run experiments using:
```
python run_experiments.py -c <config_file> -e <exp_name> -n <config_num>
```
The exp_name is a prefix for the filenames of CSV files where stats for the experiments are recorded.
Each of the command line arguments has defaults. Please refer to the documentation inside [`run_experiments.py`] for further details. (Or run it with the `-h` flag.)

The config files for experiments from the [paper](https://arxiv.org/abs/1909.07750) are in the experiments directory.
The name of the file corresponding to an experiment is formed as: <ALGORITHM_NAME>_<META_FEATURE_NAMES>.py

The ALGORITHM_NAMEs we used were: dqn, rainbow, a3c and a3c_lstm
The META_FEATURE_NAMES we used were: seq_del (for delay and sequence length varied together), noises (for P and R noises varied together), sparsity (for varying make_denser) and sparsity_2 (for varying reward density)
For example, for algorithm DQN when varying meta-features delay and sequence length, the corresponding experiment file is custom_agents_dqn_seq_del.py



It will generate 2 CSV files: file_name_prefix.csv and file_name_prefix_eval.csv
The 1st one stores stats for training runs and the 2nd one for evaluation runs.

To plot results from these, run jupyter-notebook and open plot_experiments.ipynb in Jupyter. There are instructions within each of the cells on how to generate the plots from the paper.

#### NOTE: For Rainbow, due to some memory leak bug (possibly for TensorFlow), we had to split the experiments into multiple files for seq_del and noises experiments.

If you have additional questions, please feel free to open an issue on the anonymous GitHub repository.


### ADDITIONAL INFO:
The file -MDP-Playground/rl_toy/envs/rl_toy_env.py contains the code for the Environments. It can be run as python rl_toy_env.py to instantiate Environments and run a random agent on them.


### HOW TO RUN FURTHER EXPERIMENTS:
The file run_experiments.py can be used to run further configurations that were not run in the paper. It has by default the hyperparameter values for the DQN experiments in the paper. To use another algorithm, you would need to change the hyperparameters to be the ones for the desired algorithm. The config for the meta-features, etc. would need to be given in a Python config file. See default_config.py for an example config file.

It can be run as "python3 run_experiments.py --config-file <config_file> --output-file <file_name_prefix CSV for CSV files where stats are saved>".

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
