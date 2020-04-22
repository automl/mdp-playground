# MDP Playground

#TODO Keep the next block or the one after it (that one is for the anonips repo)
# To install all requirements:
conda env create -n <env_name> -f py36_toy_rl.yml
conda activate <env_name>
git clone git@github.com:RaghuSpaceRajan/custom-gym-env.git
cd custom-gym-env
pip install -e .
git clone git@github.com:RaghuSpaceRajan/gym-extension.git
cd gym-extension
pip install -e .

# To install all requirements:
conda env create -n <env_name> -f py36_toy_rl.yml

conda activate <env_name>

git clone https://github.com/anonips/-MDP-Playground.git

cd ./-MDP-Playground

pip install -e .

git clone https://github.com/anonips/Gym-Extension.git

cd Gym-Extension

pip install -e .



## HOW TO RUN EXPERIMENTS:
The experiments for the paper are in the experiments directory. The name of the file corresponding to an experiment is formed as: custom_agents_<ALGORITHM_NAME>_<META_FEATURE_NAMES>.py
The ALGORITHM_NAMEs we used were: dqn, rainbow, a3c and a3c_lstm
The META_FEATURE_NAMES we used were: seq_del (for delay and sequence length varied together), noises (for P and R noises varied together), sparsity (for varying make_denser) and sparsity_2 (for varying reward density)
For example, for algorithm DQN when varying meta-features delay and sequence length, the corresponding experiment file is custom_agents_dqn_seq_del.py

You can run experiments using: python <experiment_file> <file_name_prefix for CSV files where stats are saved>

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
If the above procedure for install didn't work (PyYAML gave us errors sometimes for the above procedure):
conda create -n <env_name> python=3.6

pip install ray[rllib,debug]==0.7.3

pip install tensorflow==1.13.0rc1

pip install pandas==0.25.0

pip install requests==2.22.0

git clone https://github.com/anonips/-MDP-Playground.git

cd -MDP-Playground

pip install -e .

git clone https://github.com/anonips/Gym-Extension.git

cd Gym-Extension

pip install -e .
