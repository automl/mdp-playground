import numpy as np
import pandas as pd 
import argparse, os
import json

class MDPPToCave():
    def __init__(self, output_dir = "../mdpp_to_cave/"):
        self.output_folder = output_dir
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _create_configspace_json(self, stats_pd, var_configs):
        configspace = {"hyperparameters":[],
                    "conditions": [],
                    "forbiddens": [],
                    "python_module_version": "0.4.11",
                    "json_format_version": 0.1}
        for param in var_configs:
            param_config = {"name": param,
                            "type": "uniform_int",
                            }
            var_type = str( type(stats_pd[param].iloc[0]) )    
            if("int" in var_type or "bool" in var_type):
                param_config["lower"] = int( stats_pd[param].min() )
                param_config["upper"] = int( stats_pd[param].max() )
                param_config["default"] = int(param_config["lower"] + param_config["upper"]//2)
                param_config["type"] = "uniform_int"
            else:#Float
                param_config["lower"] = float( stats_pd[param].min() )
                param_config["upper"] = float( stats_pd[param].max() )
                param_config["default"] = (param_config["lower"] + param_config["upper"])//2
                param_config["type"] = "uniform_float"
            
            if "lr" in param:
                param_config["log"] = True
            else:
                param_config["log"] = False
            configspace["hyperparameters"].append(param_config)
        return configspace

    def _create_run_history(self, stats_pd, seed_idx, col_names, output_folder, var_configs):
        final_rows_for_a_config = []
        for i in range(stats_pd.shape[0] - 1):
            if stats_pd["timesteps_total"].iloc[i] > stats_pd["timesteps_total"].iloc[ i + 1]:
                final_rows_for_a_config.append(i)
        final_rows_for_a_config.append(stats_pd.shape[0]-1) # Always append the last row!

        runhistory_col_names = ["cost","time","status","budget","seed"]
        runhistory_df = pd.DataFrame(columns=runhistory_col_names)
        runhistory_df["cost"] = -1 * stats_pd["episode_reward_mean"].iloc[final_rows_for_a_config]
        runhistory_df["time"] = stats_pd["episode_len_mean"].iloc[final_rows_for_a_config]
        runhistory_df["budget"] = stats_pd["timesteps_total"].iloc[final_rows_for_a_config]
        if seed_idx > 0:
            runhistory_df["seed"] = stats_pd[col_names[seed_idx]].iloc[final_rows_for_a_config]
        else:
            runhistory_df["seed"] = 0
        runhistory_df["status"] = "SUCCESS"
        for var in var_configs:
            runhistory_df[var] = stats_pd[var].iloc[final_rows_for_a_config]
        return runhistory_df

    def to_cave_csv(self, args):
        #file_path = args.file_path
        input_folder = "../mdp_files/"
        file_name = "dqn_vanilla_learning_starts"
        output_folder = "../to_cave_format/%s"%file_name
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        ## Read current csvs ##
        stats_file = os.path.join(input_folder, file_name)
        with open(stats_file + '.csv') as file_:
            col_names = file_.readline().strip().split(', ')
            col_names[0] = col_names[0][2:] # to remove '# ' that was written
        # print("config_names:", col_names)
        stats_pd = pd.read_csv(stats_file + '.csv', skip_blank_lines=True,\
                                    header=None, names = col_names, comment='#', sep=' ')
        remove_names = ["training_iteration", "algorithm", "seed"]
        
        parameters = col_names[:-3].copy()#All paramaters tracked in run
        for x in col_names:
            for name in remove_names:
                if(name in x):
                    parameters.remove(x)

        #Compute parameters that varied and store value in dict
        config_values = {}
        seed_idx = -1
        for i, c in enumerate(col_names): # hardcoded 3 for no. of stats written
            if(c in parameters): #parameters we care about
                config_values[c] = stats_pd[c].unique() #values a given parameter took
            if("seed" in c):
                seed_idx = i
        var_configs = [p for p in parameters if len(config_values[p])>1]

        configspace = self._create_configspace_json(stats_pd, var_configs)
        output_configspace = os.path.join(output_folder,'configspace.json')
        with open(output_configspace, 'w') as fp:
            json.dump(configspace, fp, indent=2)
        
        scenario_str = "paramfile = ./configspace.json\nrun_obj = quality"
        output_configspace = os.path.join(output_folder,'scenario.txt')
        with open(output_configspace, 'w') as fp:
            fp.write(scenario_str)

        #Runhistory and trajectory files
        runhistory_df = self._create_run_history(stats_pd, seed_idx, col_names, output_folder, var_configs)
        runhistory_df.to_csv( os.path.join(output_folder,'runhistory.csv'), header=True, index=False)

    ## Creates the bohb file from the mdpp output in the output_folder directory
    ## this file can be fed into cave for analysis
    def to_bohb_results(self, dir_name, exp_name):
        #file_path = args.file_path
        output_folder = os.path.join(self.output_folder, exp_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        ## Read current csvs ##
        stats_file = os.path.join(dir_name, exp_name)
        with open(stats_file + '.csv') as file_:
            col_names = file_.readline().strip().split(', ')
            col_names[0] = col_names[0][2:] # to remove '# ' that was written
        # print("config_names:", col_names)
        stats_pd = pd.read_csv(stats_file + '.csv', skip_blank_lines=True,\
                                    header=None, names = col_names, comment='#', sep=' ')
        remove_names = ["training_iteration", "algorithm", "seed"]
        
        parameters = col_names[:-3].copy()#All paramaters tracked in run
        for x in col_names:
            for name in remove_names:
                if(name in x):
                    parameters.remove(x)

        #Compute parameters that varied and store value in dict
        config_values = {}
        seed_idx = -1
        for i, c in enumerate(col_names): # hardcoded 3 for no. of stats written
            if(c in parameters): #parameters we care about
                config_values[c] = stats_pd[c].unique() #values a given parameter took
            if("seed" in c):
                seed_idx = i
        var_configs = [p for p in parameters if len(config_values[p])>1]

        final_rows_for_a_config = []
        for i in range(stats_pd.shape[0] - 1):
            if stats_pd["timesteps_total"].iloc[i] > stats_pd["timesteps_total"].iloc[ i + 1]:
                final_rows_for_a_config.append(i)
        final_rows_for_a_config.append(stats_pd.shape[0]-1) # Always append the last row!

        ##------------- Start converting csv ----------------##
        #configspace and scenario file
        configspace = self._create_configspace_json(stats_pd, var_configs)
        output_configspace = os.path.join(output_folder,'configspace.json')
        with open(output_configspace, 'w') as fp:
            json.dump(configspace, fp, indent=2)

        #Trajectory and runhistory files
        #Finding end configuration training
        diff_configs = stats_pd.iloc[final_rows_for_a_config]
        diff_configs = diff_configs.groupby(var_configs)
        configs_mean = diff_configs.mean() 
        diff_configs_results = [] #results.json
        diff_configs_lst = []
        budget = stats_pd["timesteps_total"].iloc[final_rows_for_a_config[0]]#all have the same budget
        aux = 0
        for i, group_name in enumerate(diff_configs.groups):
            group_labels = diff_configs.groups[group_name]
            config_id = [0, 0, i]
            config_dict = {}
            #configs.json
            config_lst = [config_id]
            for name in var_configs:
                config_dict[name] = stats_pd[name].iloc[group_labels[0]].item()
            config_lst.append(config_dict)
            config_lst.append({"model_based_pick": False})
            diff_configs_lst.append(config_lst)

            #results.json
            mean_reward = configs_mean["episode_reward_mean"].iloc[i]#mean along seed
            results_lst=[config_id, budget.item(), {"submitted": float("%.2f"%aux),
                                                    "started": float("%.2f"%(aux + 0.1)),
                                                    "finished": float("%.2f"%(aux + 1)),} ]
            aux += 1.1
            results_dict = {"loss": -mean_reward.item(), 
                            "info": {}}
            results_lst.append(results_dict)
            results_lst.append(None)
            diff_configs_results.append(results_lst)

        #configs.json
        output_configs = os.path.join(output_folder,'configs.json')
        with open(output_configs, 'w') as fout:
            for d in diff_configs_lst:
                json.dump(d, fout)
                fout.write('\n')

        #results.json
        output_configs = os.path.join(output_folder,'results.json')
        with open(output_configs, 'w') as fout:
            for d in diff_configs_results:
                json.dump(d, fout)
                fout.write('\n')
        return output_folder
                
if __name__ == "__main__":
    dir_name = '../../../mdp_files/'
    exp_name = 'dqn_seq_del'
    
    from cave.cavefacade import CAVE
    import os
    #The converted mdpp csvs will be stored in output_dir
    output_dir = "../../../mdpp_to_cave"
    mdpp_file =  os.path.join(dir_name, exp_name)
    mdpp_cave = MDPPToCave(output_dir)
    cave_input_file = mdpp_cave.to_bohb_results(dir_name, exp_name)

    cave_input_file = "../../../mdpp_to_cave/dqn_seq_del"

    # Similarly, as an example, cave will ouput it's results 
    # to the same directory as cave's input files

    cave_results = os.path.join(cave_input_file, "output")
    print(os.path.abspath(cave_results))
    cave = CAVE(folders = [cave_input_file],
                output_dir = cave_results,
                ta_exec_dir = [cave_input_file],
                file_format = "BOHB",
                show_jupyter=True,
    )
    
    #cave.plot_scatter()
    cave.pimp_forward_selection()