import numpy as np
import pandas as pd
import argparse, os
import json
from mdp_playground.analysis import MDPP_Analysis
from cave.cavefacade import CAVE


class MDPPToCave:
    def __init__(self):
        return

    def _create_configspace_json(self, stats_pd, var_configs):
        configspace = {
            "hyperparameters": [],
            "conditions": [],
            "forbiddens": [],
            "python_module_version": "0.4.11",
            "json_format_version": 0.1,
        }
        for param in var_configs:
            param_config = {"name": param}
            var_type = str(type(stats_pd[param].iloc[0]))
            if "int" in var_type or "bool" in var_type:
                param_config["lower"] = int(stats_pd[param].min())
                param_config["upper"] = int(stats_pd[param].max())
                param_config["default"] = int(
                    param_config["lower"] + param_config["upper"] // 2
                )
                param_config["type"] = "uniform_int"
            elif "str" in var_type:
                # Categorical
                param_config["type"] = "categorical"
                param_config["choices"] = list(stats_pd["conv_filters"].unique())
                param_config["default"] = param_config["choices"][0]
            else:  # Float
                param_config["lower"] = float(stats_pd[param].min())
                param_config["upper"] = float(stats_pd[param].max())
                param_config["default"] = (
                    param_config["lower"] + param_config["upper"]
                ) / 2
                param_config["type"] = "uniform_float"

            if "lr" in param:
                param_config["log"] = True
            else:
                param_config["log"] = False
            configspace["hyperparameters"].append(param_config)
        return configspace

    def _create_run_history(
        self, stats_pd, seed_idx, col_names, output_folder, var_configs
    ):
        final_rows_for_a_config = []
        for i in range(stats_pd.shape[0] - 1):
            if (
                stats_pd["timesteps_total"].iloc[i]
                > stats_pd["timesteps_total"].iloc[i + 1]
            ):
                final_rows_for_a_config.append(i)
        final_rows_for_a_config.append(
            stats_pd.shape[0] - 1
        )  # Always append the last row!

        runhistory_col_names = ["cost", "time", "status", "budget", "seed"]
        runhistory_df = pd.DataFrame(columns=runhistory_col_names)
        runhistory_df["cost"] = (
            -1 * stats_pd["episode_reward_mean"].iloc[final_rows_for_a_config]
        )
        runhistory_df["time"] = stats_pd["episode_len_mean"].iloc[
            final_rows_for_a_config
        ]
        runhistory_df["budget"] = stats_pd["timesteps_total"].iloc[
            final_rows_for_a_config
        ]
        if seed_idx > 0:
            runhistory_df["seed"] = stats_pd[col_names[seed_idx]].iloc[
                final_rows_for_a_config
            ]
        else:
            runhistory_df["seed"] = 0
        runhistory_df["status"] = "SUCCESS"
        for var in var_configs:
            runhistory_df[var] = stats_pd[var].iloc[final_rows_for_a_config]
        return runhistory_df

    def join_files(self, file_prefix, file_suffix):
        """Utility to join files that were written with different experiment configs"""
        with open(file_prefix + file_suffix, "ab") as combined_file:
            i = 0
            num_diff_lines = []
            while True:
                if os.path.isfile(file_prefix + "_" + str(i) + file_suffix):
                    with open(
                        file_prefix + "_" + str(i) + file_suffix, "rb"
                    ) as curr_file:
                        byte_string = curr_file.read()
                        newline_count = byte_string.count(10)
                        num_diff_lines.append(newline_count)
                        combined_file.write(byte_string)
                else:
                    break
                i += 1
            print(
                str(i)
                + " files were combined into 1 for file:"
                + file_prefix
                + file_suffix
            )
            # print("Files missing for config_nums:", missing_configs, ". Did you pass the right value for max_total_configs as an argument?")
            # print("Unique line count values:", np.unique(num_diff_lines))
            if i == 0:
                raise FileNotFoundError(
                    "No files to combine were present. Please check your location and/or filenames that they are correct."
                )

    def _read_stats(self, stats_file):
        if os.path.isfile(stats_file + ".csv"):
            print(
                "\033[1;31mLoading data from a sequential run/already combined runs of experiment configurations.\033[0;0m"
            )
        else:
            print(
                "\033[1;31mLoading data from a distributed run of experiment configurations. Creating a combined CSV stats file.\033[0;0m"
            )
            self.join_files(stats_file, ".csv")
            self.join_files(stats_file, "_eval.csv")

    def to_cave_csv(self, args):
        # file_path = args.file_path
        input_folder = "../mdp_files/"
        file_name = "dqn_vanilla_learning_starts"
        output_folder = "../to_cave_format/%s" % file_name
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ## Read current csvs ##
        stats_file = os.path.join(input_folder, file_name)
        with open(stats_file + ".csv") as file_:
            col_names = file_.readline().strip().split(", ")
            col_names[0] = col_names[0][2:]  # to remove '# ' that was written
        # print("config_names:", col_names)
        stats_pd = pd.read_csv(
            stats_file + ".csv",
            skip_blank_lines=True,
            header=None,
            names=col_names,
            comment="#",
            sep=" ",
        )
        remove_names = ["training_iteration", "algorithm", "seed"]

        parameters = col_names[:-3].copy()  # All paramaters tracked in run
        for x in col_names:
            for name in remove_names:
                if name in x:
                    parameters.remove(x)

        # Compute parameters that varied and store value in dict
        config_values = {}
        seed_idx = -1
        for i, c in enumerate(col_names):  # hardcoded 3 for no. of stats written
            if c in parameters:  # parameters we care about
                config_values[c] = stats_pd[c].unique()  # values a given parameter took
            if "seed" in c:
                seed_idx = i
        var_configs = [p for p in parameters if len(config_values[p]) > 1]

        configspace = self._create_configspace_json(stats_pd, var_configs)
        output_configspace = os.path.join(output_folder, "configspace.json")
        with open(output_configspace, "w") as fp:
            json.dump(configspace, fp, indent=2)

        scenario_str = "paramfile = ./configspace.json\nrun_obj = quality"
        output_configspace = os.path.join(output_folder, "scenario.txt")
        with open(output_configspace, "w") as fp:
            fp.write(scenario_str)

        # Runhistory and trajectory files
        runhistory_df = self._create_run_history(
            stats_pd, seed_idx, col_names, output_folder, var_configs
        )
        runhistory_df.to_csv(
            os.path.join(output_folder, "runhistory.csv"), header=True, index=False
        )

    def to_bohb_results(
        self, input_dir, exp_name, output_dir="../cave_output/", overwrite=False
    ):
        """Converts MDP Playground stats CSVs to BOHB format stats files:
        configs.json, results.json, configspace.json, in output_dir/exp_name.
        This file can be fed into cave for further analysis.

        Currently only compatible with the MDPP expt. of type: grid of configs

        exp_name : str
            Should be the expt name from MDPP, i.e., the "prefix" of the CSV stats files. A sub-directory of output_dir is created with this name to store BOHB format stats files.

        overwrite : bool
            If existing files should be overwritten.

        Returns "<output_dir>/<exp_name>"
        """

        print("Writing BOHB to cave output to %s" % (os.path.abspath(output_dir)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # file_path = args.file_path
        output_dir_final = os.path.join(output_dir, exp_name)
        if not os.path.exists(output_dir_final):
            os.makedirs(output_dir_final)

        # Read current CSVs

        ##TODO Re-use code from analyis.py to load data instead of processing it again here:
        # mdpp_analysis = MDPP_Analysis()
        # self.exp_data = mdpp_analysis.get_exp_data(dir_name=input_dir,
        #                             exp_name=exp_name,
        #                             )
        # print("exp_data:\n", self.exp_data)

        # exp_data["dims_varied"]
        # exp_data["dims_values"]

        stats_file = os.path.join(input_dir, exp_name)
        stats_file = os.path.abspath(stats_file)
        self._read_stats(stats_file)
        with open(stats_file + ".csv") as file_:
            col_names = file_.readline().strip().split(", ")
            col_names[0] = col_names[0][2:]  # to remove '# ' that was written
        # print("config_names:", col_names)
        stats_pd = pd.read_csv(
            stats_file + ".csv",
            skip_blank_lines=True,
            header=None,
            names=col_names,
            comment="#",
            sep=" ",
        )
        remove_names = ["training_iteration", "algorithm", "seed"]

        parameters = col_names[:-3].copy()  # All parameters tracked in run
        for x in col_names:
            for name in remove_names:
                if name in x:
                    parameters.remove(x)

        # Compute parameters that varied and store value in dict
        config_values = {}
        seed_idx = -1
        for i, c in enumerate(col_names):  # hardcoded 3 for no. of stats written
            if c in parameters:  # parameters we care about
                config_values[c] = stats_pd[c].unique()  # values a given parameter took
            if "seed" in c:
                seed_idx = i
        var_configs = [p for p in parameters if len(config_values[p]) > 1]

        final_rows_for_a_config = []
        for i in range(stats_pd.shape[0] - 1):
            if (
                stats_pd["timesteps_total"].iloc[i]
                > stats_pd["timesteps_total"].iloc[i + 1]
            ):
                final_rows_for_a_config.append(i)
        final_rows_for_a_config.append(
            stats_pd.shape[0] - 1
        )  # Always append the last row!

        ##------------- Start converting csv ----------------##
        # configspace and scenario file
        configspace = self._create_configspace_json(stats_pd, var_configs)
        cs_json_file = os.path.join(output_dir_final, "configspace.json")
        if os.path.exists(cs_json_file):
            if not overwrite:
                raise FileExistsError()

        with open(cs_json_file, "w") as fp:
            json.dump(configspace, fp, indent=2)

        # print("var_configs:", var_configs)

        # Trajectory and runhistory files
        # Finding end configuration training
        diff_configs = stats_pd.iloc[final_rows_for_a_config]
        # print("diff_configs:", diff_configs)
        diff_configs = diff_configs.groupby(var_configs)
        # print("grouped by", diff_configs)
        configs_mean = diff_configs.mean()
        # print("mean:", configs_mean)
        # print("diff_configs.groups:", diff_configs.groups)
        diff_configs_results = []  # results.json
        diff_configs_lst = []
        budget = stats_pd["timesteps_total"].iloc[
            final_rows_for_a_config[0]
        ]  # all have the same budget
        aux = 0
        for i, group_name in enumerate(diff_configs.groups):
            group_labels = diff_configs.groups[group_name]
            config_id = [0, 0, i]
            config_dict = {}
            # configs.json
            config_lst = [config_id]
            for name in var_configs:
                value = stats_pd[name].iloc[group_labels[0]]
                if isinstance(value, str):
                    config_dict[name] = value
                else:
                    config_dict[name] = value.item()
            config_lst.append(config_dict)
            config_lst.append({"model_based_pick": False})
            diff_configs_lst.append(config_lst)

            # results.json
            mean_reward = configs_mean["episode_reward_mean"].iloc[i]  # mean along seed
            results_lst = [
                config_id,
                budget.item(),
                {
                    "submitted": float("%.2f" % aux),
                    "started": float("%.2f" % (aux + 0.1)),
                    "finished": float("%.2f" % (aux + 1)),
                },
            ]
            aux += 1.1
            results_dict = {"loss": -mean_reward.item(), "info": {}}
            results_lst.append(results_dict)
            results_lst.append(None)
            diff_configs_results.append(results_lst)

        # configs.json
        output_configs = os.path.join(output_dir_final, "configs.json")
        if os.path.exists(output_configs):
            if not overwrite:
                raise FileExistsError()
        with open(output_configs, "w") as fout:
            for d in diff_configs_lst:
                json.dump(d, fout)
                fout.write("\n")

        # results.json
        output_results = os.path.join(output_dir_final, "results.json")
        if os.path.exists(output_results):
            if not overwrite:
                raise FileExistsError()
        with open(output_results, "w") as fout:
            for d in diff_configs_results:
                json.dump(d, fout)
                fout.write("\n")

        return output_dir_final

    def to_CAVE_object(
        self, input_dir, exp_name, output_dir="../cave_output/", overwrite=False
    ):
        """Converts MDP Playground stats CSVs to BOHB format stats files and creates
        a CAVE object from them.

        Please see to_bohb_results() for details about some of the parameters.
        """

        cave_input_file = self.to_bohb_results(
            input_dir, exp_name, output_dir, overwrite=overwrite
        )

        cave_results = os.path.join(cave_input_file, "out")
        cave = CAVE(
            folders=[cave_input_file],
            output_dir=cave_results,
            ta_exec_dir=[cave_input_file],
            file_format="BOHB",
            show_jupyter=True,
        )

        return cave


if __name__ == "__main__":
    input_dir = "../mdp_files/"
    exp_name = "dqn_seq_del"

    from cave.cavefacade import CAVE
    import os

    # The converted mdpp csvs will be stored in output_dir
    output_dir = "../mdpp_to_cave"
    mdpp_file = os.path.join(input_dir, exp_name)
    mdpp_cave = MDPPToCave()
    cave_input_file = mdpp_cave.to_bohb_results(input_dir, exp_name, output_dir)

    # cave_input_file = "../../../mdpp_to_cave/dqn_seq_del"

    # Similarly, as an example, cave will ouput it's results
    # to the same directory as cave's input files

    cave_results = os.path.join(cave_input_file, "output")
    print(os.path.abspath(cave_results))
    cave = CAVE(
        folders=[cave_input_file],
        output_dir=cave_results,
        ta_exec_dir=[cave_input_file],
        file_format="BOHB",
        show_jupyter=True,
    )

    # Common analysis
    cave.performance_table()
    cave.local_parameter_importance()
    cave.cave_fanova()  # can only be used with more than 1 meta-feature

    # Other analysis
    # cave.parallel_coordinates()
    # cave.cost_over_time()
    # cave.algorithm_footprints()
    # cave.pimp_comparison_table()
    # cave.cave_ablation()
    # cave.pimp_forward_selection()
    # cave.feature_importance()
    # cave.configurator_footprint()
    # cave.algorithm_footprints()
    # cave.plot_ecdf()
    # cave.plot_scatter()
    # cave.compare_default_incumbent()
    # cave.overview_table()
