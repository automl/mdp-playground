import sys, os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__) # docstring at beginning of the file is stored in __doc__
    parser.add_argument('-c', '--config-file', dest='config_file', action='store', default='default_config',
                    help='Configuration file containing configuration to run experiments. It must be a Python file so config can be given programmatically. There are 2 types of configs - VARIABLE CONFIG across the experiments and STATIC CONFIG across the experiments. \nVARIABLE CONFIGS: The OrderedDicts var_env_configs, var_agent_configs and var_model_configs hold configuration options that are variable for the environment, agent and model across the current experiment. For each configuration option, the option is the key in the dict and its value is a list of values it can take for the current experiment.  A Cartesian product of these lists is taken to generate various possible configurations to be run. For example, you might want to vary "delay" for the current experiment. Then "delay" would be a key in var_env_configs dict and its value would be a list of values it can take. Because Ray does not have a common way to address this specification of configurations for its agents, there are a few hacky ways to set var_agent_configs and var_model_configs currently. Please see sample experiment config files in the experiments directory to see how to set the values for a given algorithm. \nSTATIC CONFIGS: env_config, agent_config and model_config are dicts which hold the static configuration for the current experiment as a normal Python dict.')
    parser.add_argument('-e', '--exp-name', dest='exp_name', action='store', default='mdpp_default_experiment',
                    help='The user-chosen name of the experiment. This is used as the prefix of the output files (the prefix also contains config_num if that is provided). It will save stats to 2 CSV files, with the filenames as the one given as argument'
                    ' and another file with an extra "_eval" in the filename that contains evaluation stats during the training. Appends to existing files or creates new ones if they don\'t exist.')
    parser.add_argument('-n', '--config-num', dest='config_num', action='store', default=None, type=int,
                    help='Used for running the configurations of experiments in parallel. This is appended to the prefix of the output files (after exp_name).'
                    ' A Cartesian product of different configuration values for the experiment will be taken and ordered as a list and this number corresponds to the configuration number in this list.'
                    ' Please look in to the code for details.')
    parser.add_argument('-fwk', '--framework', dest='framework', action='store', type=str,
                    help='Used to select the framework in which the experiments will be run, default value is ray. Options: ray, baselines')

    args = parser.parse_args()
    print("Parsed args:", args)
    return args

def main():
    args = parse_args()
    if args.framework is not None: # get from arguments
        framework =  args.framework
    else: #check config file
        if args.config_file[-3:] == '.py':
            args.config_file = args.config_file[:-3]
        config_file_path = os.path.abspath('/'.join(args.config_file.split('/')[:-1]))
        sys.path.insert(1, config_file_path)
        config = importlib.import_module(args.config_file.split('/')[-1], package=None)
        try:
            framework = config.framework
        except AttributeError:
            #Framework not defined either in args or config file
            #set default value
            framework = "ray"

    #Run file
    print("Running framework:", framework)
    if(framework == "stable_baselines"):
        import run_experiments_baselines
        run_experiments_baselines.main(args)
    else: #ray
        import run_experiments#_ray
        run_experiments.main(args)#_ray.main(args)


if __name__ == '__main__':
    main()