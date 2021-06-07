import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import radar_chart
import os
import warnings
import math


class MDPP_Analysis:
    """Utility class to load and plot data for analysis of experiments from MDP Playground"""

    def __init__(self):
        pass

    def load_data(self, experiments: dict, load_eval=True, exp_type="grid"):
        """Loads training and evaluation data from given multiple files

        Parameters
        ----------
        experiments : dict<str,str>
            per experiment -> key-value pair of exp_name & dir_name

            dir_name : str
                The location where the training and evaluation CSV files were written
            exp_name : str
                The name of the experiment: the training and evaluation CSV filenames are formed using this string

            eg: experiments = {
                    "td3_move_to_a_point_action_max": "<path_to_data_dir>",
                    .....
                }

        Returns
        -------
            list of experiment data of type <dict>
            [ for more details refer get_exp_data() ]

        """
        list_exp_data = []
        for exp_name, dir_name in experiments.items():
            exp_data = self.get_exp_data(
                dir_name, exp_name, exp_type=exp_type, load_eval=load_eval
            )
            list_exp_data.append(exp_data)
        return list_exp_data

    def get_exp_data(
        self,
        dir_name,
        exp_name,
        exp_type="grid",
        num_metrics=3,
        load_eval=True,
        threshold=0.05,
        sample_freq=1,
    ):
        # , max_total_configs=200):
        """Get training and evaluation data from a single set of recorded CSV stats files.

        Parameters
        ----------
        dir_name : str
            The location where the training and evaluation CSV files were written
        exp_name : str
            The name of the experiment: the training and evaluation CSV filenames are formed using this string
        exp_type : str
            One of ['grid', 'random']. If it's 'grid', it's assumed that a grid of configurations was run and data loading takes place in a manner specific to grids. Otherwise, data loading tries to look for different unique configurations run.
        num_metrics : int
            The number of metrics that were written to CSV stats files. Default is 3 (timesteps_total, episode_reward_mean, episode_len_mean).
        load_eval : bool
            Whether to load evaluation stats CSV or not.
        threshold : float
            The fault tolerance threshold while loading data. Show warnings for files where the number of data rows deviates by more than the threshold value (from the mean number of rows across the loaded files), e.g., if the threshold = 0.05 and mean number of rows across the loaded files = 20, a warning is displayed when for a loaded file the number of rows < 19.
        sample_freq : int
            The subsampling frequency when loading data. Sub-select data lines based on this value, e.g., if sample_freq = 5, load every 5th row in a file.

        Returns
        -------
        experiment data: dictionary type with following key-value

        train_stats : np.ndarray
            Training stats at end of training: 8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        eval_stats : np.ndarray
            Training stats at end of training: 8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        train_curves: np.ndarray
            The loaded training CSV with the last 3 columns the train stats that were saved and the initial columns are various setting for the algorithm and environment.
        eval_curves: np.ndarray
            The loaded evaluation CSV with the columns the evaluation stats that were saved
        """

        stats_file = (
            dir_name + "/" + exp_name
        )  # Name of file to which benchmark stats were written
        self.stats_file = stats_file

        if os.path.isfile(stats_file + ".csv"):
            print(
                "\033[1;31mLoading data from a sequential run/already combined runs of experiment configurations:\033[0;0m "
                + stats_file
                + ".csv"
            )
        else:
            print(
                "Loading data from a distributed run of experiment configurations. Creating a combined CSV stats file."
            )
            train_data = dict()
            eval_data = dict()

            def join_files(file_prefix, file_suffix):
                """Utility to join files that were written with different experiment configs"""
                with open(file_prefix + file_suffix, "ab") as combined_file:
                    i = 0
                    missing_configs = []
                    num_diff_lines = []
                    while True:  # i < max_total_configs:
                        if os.path.isfile(file_prefix + "_" + str(i) + file_suffix):
                            with open(
                                file_prefix + "_" + str(i) + file_suffix, "rb"
                            ) as curr_file:
                                byte_string = curr_file.read()
                                newline_count = byte_string.count(10)
                                num_diff_lines.append(newline_count)
                                combined_file.write(byte_string)

                                if file_suffix == ".csv":
                                    # train data
                                    train_data[
                                        os.path.basename(curr_file.name)
                                    ] = newline_count
                                else:
                                    # eval data
                                    eval_data[
                                        os.path.basename(curr_file.name)
                                    ] = newline_count
                        else:
                            # missing_configs.append(i)
                            break
                        i += 1
                    print(
                        str(i)
                        + " files were combined into 1 for file:"
                        + file_prefix
                        + "_n"
                        + file_suffix
                    )
                    # print("Files missing for config_nums:", missing_configs, ". Did you pass the right value for max_total_configs as an argument?")
                    # print("Unique line count values:", np.unique(num_diff_lines))
                    if i == 0:
                        raise FileNotFoundError(
                            "No files to combine were present. Please check your location and/or filenames that they are correct. Filename passed: " +
                            file_prefix +
                            file_suffix)

            join_files(stats_file, ".csv")
            join_files(stats_file, "_eval.csv")

            train_mean_cnt = np.mean(list(train_data.values()))
            eval_mean_cnt = np.mean(list(eval_data.values()))

            # fault tolerance check
            for file_name, line_count in train_data.items():
                if line_count < train_mean_cnt * (1 - threshold):
                    warnings.warn(
                        "Expected a minimum of {0} rows in each stats file. Got only: {1} in file: {2}".format(
                            train_mean_cnt * (1 - threshold), line_count, file_name
                        )
                    )
            for file_name, line_count in eval_data.items():
                if line_count < eval_mean_cnt * (1 - threshold):
                    warnings.warn(
                        "Expected a minimum of {0} rows in each stats file. Got only: {1} in file: {2}".format(
                            eval_mean_cnt * (1 - threshold), line_count, file_name
                        )
                    )

        # Read column names
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
        # subsampling code
        stats_pd_indices = np.arange(0, stats_pd.shape[0], step=sample_freq)
        stats_pd = stats_pd.loc[stats_pd_indices]

        self.stats_pd = stats_pd
        print("Training stats read (rows, columns):", stats_pd.shape)

        config_counts = []
        dims_values = []
        # Keep only config_names that we want to measure
        # traning iteration is always first, metrics are always last.
        self.full_config_names = col_names.copy()
        full_config_names = self.full_config_names
        full_config_names.remove("training_iteration")

        # config counts includes seed
        self.seed_idx = None  # seed used to be fixed as the last, i.e.,
        # quickest varying dimension in the <experiment config>.py file's
        # config space because then all runs on a single env would be recorded
        # consecutively in the stats CSV
        self.ts_idx = None
        for i, c in enumerate(full_config_names[:-num_metrics]):
            dims_values.append(stats_pd[c].unique())
            config_counts.append(stats_pd[c].nunique())
            if "seed" in c:  # ##TODO this will just set seed index to be
                # the "last" column name with seed in it.
                self.seed_idx = i
            if c == "timesteps_total":
                self.ts_idx = i

        config_counts.append(num_metrics)  # #hardcoded number of training
        # stats that were recorded
        config_counts = tuple(config_counts)
        self.metric_names = full_config_names[-num_metrics:]
        self.config_names = full_config_names[:-num_metrics]

        # Slice into training stats and get end of training stats for
        # individual training runs in the experiment
        final_rows_for_a_config = []
        previous_i = 0
        list_of_learning_curves = []
        # cols_to_take = 8

        # Finding end configuration training
        for i in range(stats_pd.shape[0] - 1):
            if (
                stats_pd["timesteps_total"].iloc[i]
                > stats_pd["timesteps_total"].iloc[i + 1]
            ):
                final_rows_for_a_config.append(i)

        # print("i, previous_i:", i, previous_i)
        final_rows_for_a_config.append(i + 1)  # Always append the last row!
        self.final_rows_for_a_config = final_rows_for_a_config
        stats_end_of_training = stats_pd.iloc[final_rows_for_a_config]
        if exp_type == "grid":
            train_stats = stats_end_of_training.iloc[
                :, -num_metrics:
            ]  # last vals are timesteps_total, episode_reward_mean, episode_len_mean
            train_stats = np.reshape(np.array(train_stats), config_counts)
        elif exp_type == "random":
            train_stats = stats_end_of_training  # Includes config values within dataframe as opposed to only perf. metrics for the grid case above
            # train_stats = stats_end_of_training.iloc[:, -num_metrics:] # Saves space
            # by not including config values. # last vals are timesteps_total,
            # episode_reward_mean, episode_len_mean ### TODO remove

            pass
        else:
            raise ValueError("Please check exp_type passed. Was:" + exp_type)

        print("train stats shape:", train_stats.shape)

        # Calculate AUC metrics
        train_aucs = []
        for i in range(len(final_rows_for_a_config)):
            if i == 0:
                to_avg_ = stats_pd.iloc[
                    0: self.final_rows_for_a_config[i] + 1, -num_metrics:
                ]
            else:
                to_avg_ = stats_pd.iloc[
                    self.final_rows_for_a_config[i - 1]
                    + 1: self.final_rows_for_a_config[i]
                    + 1,
                    -num_metrics:,
                ]
            auc = np.mean(to_avg_, axis=0)
            train_aucs.append(auc)
            # print(auc)
        train_aucs = np.array(train_aucs)

        if exp_type == "grid":
            train_aucs = np.reshape(train_aucs, config_counts)
        elif exp_type == "random":
            train_aucs_ = train_stats.copy()
            train_aucs_.iloc[:, -num_metrics:] = train_aucs
            train_aucs = train_aucs_
            pass
        print("train_aucs.shape:", train_aucs.shape)

        # Load evaluation stats
        eval_stats, mean_data_eval, eval_aucs = None, None, None
        # load_eval = False # hack #### TODO rectify
        if load_eval:
            stats_file_eval = stats_file + "_eval.csv"
            eval_stats = np.loadtxt(stats_file_eval, dtype=float)
            # print(eval_stats, eval_stats.shape)

            i = 0
            hack_indices = []
            for line in open(stats_file_eval):

                line = line.strip()
                #    print(line)
                if line.startswith("#HACK"):
                    #         print(line, i)
                    hack_indices.append(
                        i - len(hack_indices)
                    )  # appends index of last eval in this training_iteration
                i += 1

            # print("len(hack_indices), hack_indices[:5] and [:-5]:", len(hack_indices), hack_indices[:5], hack_indices[-5:])
            if hack_indices[0] == 0:  # hack
                # hardcoded removes the 1st hack_index which is at position 0 so that
                # hack_indices_10 below doesn't begin with a -10; apparently Ray seems to
                # have changed logging for evaluation (using on_episode_end) from 0.7.3 to
                # 0.9.0
                hack_indices = hack_indices[1:]
                ray_0_9_0 = True
            else:
                ray_0_9_0 = False
            hack_indices_10 = np.array(hack_indices) - 10
            # print(hack_indices_10.shape, hack_indices_10[:5], hack_indices_10[-5:])
            # print(np.array(hack_indices[1:]) - np.array(hack_indices[:-1]))
            # print("Min:", min(np.array(hack_indices[1:]) -
            # np.array(hack_indices[:-1]))) # Some problem with Ray? Sometimes no. of
            # eval episodes is less than 10.
            final_10_evals = []
            for i in range(len(hack_indices)):
                final_10_evals.append(eval_stats[hack_indices_10[i]: hack_indices[i]])
            #     print(final_10_evals[-1])
            if ray_0_9_0:  # hack
                final_10_evals.append(
                    eval_stats[hack_indices[i]:]
                )  # appends the very last eval which begins at last hack_index for Ray 0.9.0

            final_10_evals = np.array(
                final_10_evals
            )  # has 2 columns: episode reward and episode length
            # print(final_10_evals.shape, final_10_evals)

            # final_vals = fin[final_rows_for_a_config]
            # print('final_rows_for_a_config', final_rows_for_a_config)
            # print("len(final_10_evals)", final_10_evals.shape, type(final_10_evals))
            mean_data_eval = np.mean(
                final_10_evals, axis=1
            )  # this is mean over last 10 eval episodes

            # subsampling code
            eval_stats_indices = np.arange(0, mean_data_eval.shape[0], step=sample_freq)
            mean_data_eval = mean_data_eval[eval_stats_indices]

            # Adds timesteps_total column to the eval stats which did not have them:
            mean_data_eval = np.concatenate(
                (
                    np.atleast_2d(np.array(stats_pd.iloc[:, -num_metrics])).T,
                    mean_data_eval,
                ),
                axis=1,
            )

            # 1st column is episode reward, 2nd is episode length in original
            # _eval.csv file, here it's 2nd and 3rd after prepending timesteps_total
            # column above.
            final_eval_metrics_ = mean_data_eval[final_rows_for_a_config, :]
            # print(dims_values, config_counts)

            if exp_type == "grid":
                eval_stats = np.reshape(final_eval_metrics_, config_counts)
                # print(final_eval_metrics_)
            elif exp_type == "random":
                eval_stats = final_eval_metrics_
                eval_stats = np.concatenate(
                    (
                        np.atleast_2d(np.array(train_stats.iloc[:, :-num_metrics])),
                        eval_stats,
                    ),
                    axis=1,
                )  # Includes config values within dataframe as opposed to only perf. metrics for the grid case above

            eval_stats = pd.DataFrame(eval_stats, columns=train_stats.columns)
            print("eval stats shape:", eval_stats.shape)

            # Calculate AUC metrics
            eval_aucs = []
            for i in range(len(final_rows_for_a_config)):
                if i == 0:
                    to_avg_ = mean_data_eval[
                        0: self.final_rows_for_a_config[i] + 1, -num_metrics:
                    ]
                else:
                    to_avg_ = mean_data_eval[
                        self.final_rows_for_a_config[i - 1]
                        + 1: self.final_rows_for_a_config[i]
                        + 1,
                        -num_metrics:,
                    ]
                auc = np.mean(to_avg_, axis=0)
                eval_aucs.append(auc)
                # print(auc)
            eval_aucs = np.array(eval_aucs)

            if (
                exp_type == "grid"
            ):  # TODO Do this at once for train_stats, eval_stats, train_aucs and eval_aucs
                eval_aucs = np.reshape(eval_aucs, config_counts)
            elif exp_type == "random":
                eval_aucs_ = eval_stats.copy()
                eval_aucs_.iloc[:, -num_metrics:] = eval_aucs
                eval_aucs = eval_aucs_

                pass

            print("eval_aucs.shape:", eval_aucs.shape)

        # -1 is added to ignore "no. of stats that were saved" as a dimension of difficulty
        self.config_counts = config_counts[:-1]
        self.dims_values = dims_values

        # Capture the dimensions that were varied, i.e. ones which had more than 1 value across experiments
        x_axis_labels = []
        x_tick_labels_ = []
        dims_varied = []
        for i in range(len(self.config_counts)):
            if (
                "seed" in self.config_names[i]
            ):  # ignore #seeds as dimensions of difficulty
                print("Number of seeds:", config_counts[i])
                continue
            if self.config_counts[i] > 1:
                x_axis_labels.append(self.config_names[i])
                x_tick_labels_.append([str(j) for j in self.dims_values[i]])
                for j in range(len(x_tick_labels_[-1])):
                    if len(x_tick_labels_[-1][j]) > 2:  # hack
                        abridged_str = x_tick_labels_[-1][j].split(",")
                        if abridged_str[-1] == "":
                            abridged_str = abridged_str[:-1]
                        for k in range(len(abridged_str)):
                            if abridged_str[k] == "scale":
                                abridged_str[k] = "S"
                            elif abridged_str[k] == "shift":
                                abridged_str[k] = "s"
                            elif abridged_str[k] == "rotate":
                                abridged_str[k] = "r"
                            elif abridged_str[k] == "flip":
                                abridged_str[k] = "f"
                            # abridged_str[j] = abridged_str[j][:2]
                        x_tick_labels_[-1][j] = "".join(abridged_str)
                dims_varied.append(i)

        if x_tick_labels_ == []:
            warnings.warn("No varying dims were found!")
            x_tick_labels_.append("single_config")
            x_axis_labels.append("single_config")
            dims_varied.append(0)

        self.axis_labels = x_axis_labels
        self.tick_labels = x_tick_labels_
        self.dims_varied = dims_varied
        for d, v, i in zip(x_axis_labels, x_tick_labels_, dims_varied):
            print(
                "Dimension varied:",
                d,
                ". The values it took:",
                v
                if (len(v) < 10 or not isinstance(v, list))
                else str(v[:5] + v[-5:]) + " (1st 5 and last 5)",
                ". Number of values it took:",
                config_counts[i],
                ". Index in loaded data:",
                i,
            )

        # experiment data
        exp_data = dict()
        # related to training & eval data
        exp_data["train_stats"] = train_stats
        exp_data["eval_stats"] = eval_stats
        exp_data["train_curves"] = np.array(stats_pd)
        exp_data["eval_curves"] = mean_data_eval
        exp_data["train_aucs"] = train_aucs
        exp_data["eval_aucs"] = eval_aucs

        # related to plots
        # #TODO Remove the self from these since they are per expt. variables?
        exp_data["metric_names"] = self.metric_names
        exp_data["tick_labels"] = self.tick_labels
        exp_data["axis_labels"] = self.axis_labels
        exp_data["stats_file"] = self.stats_file
        exp_data["algorithm"] = self.dims_values[0][0]
        exp_data["dims_varied"] = self.dims_varied
        exp_data["config_counts"] = self.config_counts
        exp_data["final_rows_for_a_config"] = self.final_rows_for_a_config
        exp_data["config_names"] = self.config_names
        exp_data["dims_values"] = self.dims_values
        exp_data["seed_idx"] = self.seed_idx
        exp_data["ts_idx"] = self.ts_idx

        return exp_data

    def gather_stats(self, list_exp_data, train, metric_num, plot_type, use_aucs):
        stats_data = dict()

        if plot_type == "agent":
            # groupby agent
            groupby = "algorithm"
            sub_groupby = "axis_labels"
        elif plot_type == "metric":
            # groupby metric
            groupby = "axis_labels"
            sub_groupby = "algorithm"

        # iterate and group data based on the their values eg. ['SAC', 'TD3'] or ['action_space_max', 'time_unit']
        """
        example of experiment data grouped by agent:
            {
            <group_key>
            "SAC": {
                   <sub_group_key>
                   "time_unit": {
                                "to_plot_": [],
                                "to_plot_std_": [],
                                ....
                                },
                   "action_space_max": {
                                       "to_plot_": [],
                                       "to_plot_std_": [],
                                       ....
                                       },
                   ........
                   },
            "TD3": {
                    ......
                   }
            }
        """
        for exp_data in list_exp_data:
            if train:
                if use_aucs:
                    stats = exp_data["train_aucs"]
                else:
                    stats = exp_data["train_stats"]
            else:
                if use_aucs:
                    stats = exp_data["eval_aucs"]
                else:
                    stats = exp_data["eval_stats"]

            group_keys = exp_data[groupby]
            if isinstance(group_keys, str):
                group_keys = [group_keys]

            for group_key in group_keys:
                if group_key not in stats_data:
                    # if group_key is not alreay present initialize
                    stats_data[group_key] = dict()

                sub_group_keys = exp_data[sub_groupby]
                if isinstance(sub_group_keys, str):
                    sub_group_keys = [sub_group_keys]

                for idx in range(len(sub_group_keys)):
                    sub_group_key = sub_group_keys[idx]
                    if sub_group_key not in stats_data[group_key]:
                        # if sub_group_key is not alreay present initialize
                        stats_data[group_key][sub_group_key] = dict()

                    # gather data related to plot
                    # the slice sub-selects the metric written in position metric_num from the
                    # "last axis of diff. metrics that were written" and then the axis of
                    # #seeds becomes axis=-1 ( before slice it was -2).
                    mean_data_ = np.mean(stats[..., metric_num], axis=exp_data["seed_idx"])
                    to_plot_ = np.squeeze(mean_data_)

                    # HACK traspose the array and choose first column elements always
                    if len(to_plot_.shape) > 1:
                        axes = np.arange(len(to_plot_.shape))
                        new_axes = [
                            axes[(i - idx) % len(axes)] for i, x in enumerate(axes)
                        ]
                        to_plot_ = np.transpose(to_plot_, tuple(new_axes))[:, 0]
                    stats_data[group_key][sub_group_key]["to_plot_"] = to_plot_

                    std_dev_ = np.std(
                        stats[..., metric_num], axis=exp_data["seed_idx"]
                    )  # seed
                    to_plot_std_ = np.squeeze(std_dev_)

                    # HACK traspose the array and choose first column elements always
                    if len(to_plot_std_.shape) > 1:
                        to_plot_std_ = np.transpose(to_plot_std_, tuple(new_axes))[:, 0]
                    stats_data[group_key][sub_group_key]["to_plot_std_"] = to_plot_std_

                    stats_data[group_key][sub_group_key]["labels"] = sub_group_key
                    stats_data[group_key][sub_group_key]["tick_labels"] = exp_data[
                        "tick_labels"
                    ][idx]
                    stats_data[group_key][sub_group_key]["axis_labels"] = exp_data[
                        groupby
                    ]
                    stats_data[group_key][sub_group_key]["metric_names"] = exp_data[
                        "metric_names"
                    ]
                    stats_data[group_key][sub_group_key]["stats_file"] = exp_data[
                        "stats_file"
                    ]

        return stats_data

    def plot_1d_dimensions(
        self,
        list_exp_data,
        save_fig=False,
        train=True,
        use_aucs=False,
        metric_num=-2,
        plot_type="agent",
    ):
        """Plots 1-D bar plots across a single dimension with mean and std. dev.

        Parameters
        ----------
        list_exp_data : list of experiment data of type <dict>
                   [ for more details refer load_data(), get_exp_data() ]
        save_fig : bool, optional
            A flag used to save a PDF (default is False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)
        use_aucs : bool, optional
            A flag used to insert _aucs in the filename of the PDF (default is False)
        metric_num :
            allowed values-> '-1' to plot episode mean lengths
                             '-2' to plot episode reward
        plot_type : string describing how to group data and plot, say based on agent or metric
            allowed values-> ['agent' ,'metric']

        """

        stats_data = self.gather_stats(
            list_exp_data, train, metric_num, plot_type, use_aucs
        )

        # plot
        for group_key in stats_data.keys():
            cols = 4  # hard-coded value
            rows = math.ceil(
                (len(stats_data[group_key].keys()) / cols)
            )  # dynamically compute

            plt.rcParams.update({"font.size": 18})  # default 12, for poster: 30
            plt.rcParams["figure.figsize"] = [7 * cols, 5 * rows]

            figure, axes = plt.subplots(nrows=rows, ncols=cols)  # [n*1] or [n*2] grid

            i = j = 0
            colors = [
                "blue",
                "orange",
                "green",
                "purple",
                "cyan",
                "olive",
                "brown",
                "grey",
                "red",
                "pink",
            ]
            for sub_group_key, color in zip(stats_data[group_key].keys(), colors):
                if cols == 1 and rows == 1:
                    # single row, single column plot
                    self.plot_bar(
                        axes,
                        stats_data[group_key][sub_group_key],
                        save_fig,
                        metric_num,
                        color,
                    )
                elif rows == 1:
                    # single row, multiple column plot
                    self.plot_bar(
                        axes[j],
                        stats_data[group_key][sub_group_key],
                        save_fig,
                        metric_num,
                        color,
                    )
                    j += 1
                else:
                    # multiple row, multiple column plot
                    self.plot_bar(
                        axes[i, j],
                        stats_data[group_key][sub_group_key],
                        save_fig,
                        metric_num,
                        color,
                    )
                    if j == cols - 1:
                        # switch to next row 1st column
                        j = 0
                        i += 1
                    else:
                        # switch to same row next column
                        j += 1

            # hide the blank plot (if any)
            if j > 0 and j <= cols - 1:
                while j <= cols - 1:
                    if i == 0:
                        # axes[j].set_visible(False)
                        axes[j].axis("off")
                    else:
                        # axes[i, j].set_visible(False)
                        axes[i, j].axis("off")
                    j += 1

            figure.tight_layout(pad=3.0)
            figure.suptitle(group_key, x=0.2, y=1, fontsize=24, fontweight="bold")

            # save figure
            if save_fig:
                fig_name = (
                    stats_data[group_key][sub_group_key]["stats_file"].split("/")[-1]
                    + ("_train" if train else "_eval")
                    + ("_aucs" if use_aucs else "")
                    + "_final_reward_"
                    + stats_data[group_key][sub_group_key]["axis_labels"].replace(
                        " ", "_"
                    )
                    + "_"
                    + str(
                        stats_data[group_key][sub_group_key]["metric_names"][metric_num]
                    )
                    + "_1d.pdf"
                )
                plt.savefig(fig_name, dpi=300, bbox_inches="tight")

            plt.show()

    def plot_bar(self, ax, stats_data, save_fig=False, metric_num=-2, bar_color="blue"):
        """Plots 1-D bar plots across a single dimension with mean and std. dev.

        Parameters
        ----------
        ax: matplotlib axes instance to plot
        stats_data : dictionary type with data related to ['train_stats', 'eval_stats', ...]
            [ for more details refer load_data(), get_exp_data() ]
        save_fig : bool, optional
            A flag used to save a PDF (default is False)
        metric_num :
            allowed values-> '-1' to plot episode mean lengths
                             '-2' to plot episode reward
        bar_color : the color of bars in plots
        """

        y_axis_label = (
            "Reward"
            if "reward" in stats_data["metric_names"][metric_num]
            else stats_data["metric_names"][metric_num]
        )

        to_plot_ = stats_data["to_plot_"]
        to_plot_std_ = stats_data["to_plot_std_"]
        labels = stats_data["labels"]
        tick_labels = stats_data["tick_labels"]
        axis_labels = stats_data["axis_labels"]
        stats_file = stats_data["stats_file"]

        x = np.arange(len(tick_labels))
        width = (x[1] - x[0]) * 0.8  # bar width

        ax.bar(x, to_plot_, width, yerr=to_plot_std_, label=labels, color=bar_color)

        ax.set_ylabel(y_axis_label)
        # ax.set_xlabel(axis_labels)
        ax.set_xticks(x)
        # ax.set_title(labels)
        ax.legend(loc="upper right")
        ax.set_xticklabels(tick_labels)

    def plot_2d_heatmap(self, list_exp_data, save_fig=False, train=True, metric_num=-2):
        """Plots 2 2-D heatmaps: 1 for mean and 1 for std. dev. across 2 meta-features of MDP Playground

        Parameters
        ----------
        list_exp_data : list of experiment data of type <dict>
                   [ for more details refer load_data(), get_exp_data() ]
        save_fig : bool, optional
            A flag used to save a PDF (default is False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)
        """
        # HACK
        if len(list_exp_data) > 0:
            exp_data = list_exp_data[
                0
            ]  # TODO make changes to handle multiple experiments plot
        else:
            return

        plt.rcParams.update(
            {"font.size": 18}
        )  # default 12, 24 for paper, for poster: 30
        cmap = "Purples"  # 'Blues' #
        label_ = (
            "Reward"
            if "reward" in exp_data["metric_names"][metric_num]
            else exp_data["metric_names"][metric_num]
        )

        tick_labels = exp_data["tick_labels"]
        axis_labels = exp_data["axis_labels"]
        stats_file = exp_data["stats_file"]

        if train:
            stats_data = exp_data["train_stats"]
        else:
            stats_data = exp_data["eval_stats"]

        mean_data_ = np.mean(
            stats_data[..., metric_num], axis=exp_data["seed_idx"]
        )  # seed
        to_plot_ = np.squeeze(mean_data_)
        # print(to_plot_)
        if len(to_plot_.shape) > 2:
            # warning.warn("Data contains variation in more than 2 dimensions (apart from seeds). May lead to plotting error!")
            raise ValueError(
                "Data contains variation in more than 2 dimensions (apart from seeds). This is currently not supported"
            )  # TODO Add 2-D plots for all combinations of 2 varying dims?
        plt.imshow(
            np.atleast_2d(to_plot_),
            cmap=cmap,
            interpolation="none",
            vmin=0,
            vmax=np.max(to_plot_),
        )
        if len(tick_labels) == 2:
            plt.gca().set_xticklabels(tick_labels[1])
            plt.gca().set_yticklabels(tick_labels[0])
        else:
            plt.gca().set_xticklabels(tick_labels[0])
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15  # default 15, for poster: 25
        cbar.set_label(label_, rotation=270)
        if len(axis_labels) == 2:
            plt.xlabel(axis_labels[1])
            plt.ylabel(axis_labels[0])
        else:
            plt.xlabel(axis_labels[0])
        if save_fig:
            plt.savefig(
                stats_file.split("/")[-1]
                + ("_train" if train else "_eval")
                + "_final_reward_mean_heat_map_"
                + str(exp_data["metric_names"][metric_num])
                + ".pdf",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
        std_dev_ = np.std(
            stats_data[..., metric_num], axis=exp_data["seed_idx"]
        )  # seed
        to_plot_ = np.squeeze(std_dev_)
        # print(to_plot_, to_plot_.shape)
        plt.imshow(
            np.atleast_2d(to_plot_),
            cmap=cmap,
            interpolation="none",
            vmin=0,
            vmax=np.max(to_plot_),
        )  # 60 for DQN, 100 for A3C
        if len(tick_labels) == 2:
            plt.gca().set_xticklabels(tick_labels[1])
            plt.gca().set_yticklabels([str(i) for i in tick_labels[0]])
        else:
            plt.gca().set_xticklabels(tick_labels[0])
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15  # default 15, for poster: 30
        cbar.set_label("Reward Std Dev.", rotation=270)
        if len(axis_labels) == 2:
            plt.xlabel(axis_labels[1])
            plt.ylabel(axis_labels[0])
        else:
            plt.xlabel(axis_labels[0])
        # plt.tight_layout()
        if save_fig:
            plt.savefig(
                stats_file.split("/")[-1]
                + ("_train" if train else "_eval")
                + "_final_reward_std_heat_map_"
                + str(exp_data["metric_names"][metric_num])
                + ".pdf",
                dpi=300,
                bbox_inches="tight",
            )
            # plt.savefig(stats_file.split('/')[-1] + '_train_heat_map.png')#, dpi=300)
        plt.show()

    def plot_learning_curves(
        self, list_exp_data, save_fig=False, train=True, metric_num=-2
    ):  # metric_num needs to be minus indexed because stats_pd reutrned for train stats has _all_ columns
        """Plots learning curves: Either across 1 or 2 meta-features of MDP Playground. Different colours represent learning curves for different seeds.

        Parameters
        ----------
        list_exp_data : list of experiment data of type <dict>
                   [ for more details refer load_data(), get_exp_data() ]
        save_fig : bool, optional
            A flag used to save a PDF (default is False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)
        """
        # HACK
        if len(list_exp_data) > 0:
            exp_data = list_exp_data[
                0
            ]  # TODO make changes to handle multiple experiments plot
            warnings.warn("Using only 1st expt. for the foll. plots")
        else:
            return

        stats_file = exp_data["stats_file"]
        dims_varied = exp_data["dims_varied"]
        config_counts = exp_data["config_counts"]
        config_names = exp_data["config_names"]
        dims_values = exp_data["dims_values"]
        final_rows_for_a_config = exp_data["final_rows_for_a_config"]
        if train:
            stats_data = exp_data["train_curves"]
        else:
            stats_data = exp_data["eval_curves"]

        # Plot for train metrics: learning curves; with subplot
        # Comment out unneeded labels in code lines 41-44 in this cell
        if len(dims_varied) > 1:
            ncols_ = config_counts[dims_varied[1]]
            nrows_ = config_counts[dims_varied[0]]
        else:
            ncols_ = config_counts[dims_varied[0]]
            nrows_ = 1
        nseeds_ = config_counts[exp_data["seed_idx"]]  # self.config_counts[-1]
        # print(ax, type(ax), type(ax[0]))
        # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # print("color_cycle", color_cycle)
        plt.rcParams.update({"font.size": 25})  # 25 for 36x21 fig, 16 for 24x14 fig.
        # 36x21 for better resolution but about 900kb file size, 24x14 for okay resolution and 550kb file size
        fig, ax = plt.subplots(
            nrows=nrows_, ncols=ncols_, figsize=(7 * ncols_, 5 * nrows_)
        )
        ax = np.atleast_2d(ax)
        # metrics_reshaped_squeezed = np.squeeze(metrics_reshaped)
        # print(np.squeeze(metrics_reshaped).shape)
        for i in range(len(final_rows_for_a_config)):
            i_index = i // (
                nseeds_ * ncols_
            )  # = num_seeds * shape of more frequently changing hyperparam
            j_index = (i // nseeds_) % ncols_  #
            if i == 0:
                to_plot_ = stats_data[0: final_rows_for_a_config[i] + 1, metric_num]
                to_plot_x = stats_data[
                    0: final_rows_for_a_config[i] + 1, exp_data["ts_idx"]
                ]
            else:
                to_plot_ = stats_data[
                    final_rows_for_a_config[i - 1] + 1: final_rows_for_a_config[i] + 1,
                    metric_num,
                ]
                to_plot_x = stats_data[
                    final_rows_for_a_config[i - 1] + 1: final_rows_for_a_config[i] + 1,
                    exp_data["ts_idx"],
                ]
            # print(to_plot_[-1])
            #     if i % 10 == 0:
            #         fig = plt.figure(figsize=(12, 7))
            #     print(i//50, (i//10) % 5)
            ax[i_index][j_index].plot(
                to_plot_x, to_plot_, rasterized=False
            )  # , label="Seq len" + str(seq_lens[i//10]))
            if i % nseeds_ == nseeds_ - 1:  # 10 is num. of seeds
                #         pass
                #         print("Plot no.", i//10)
                ax[i_index][j_index].set_xlabel("Train Timesteps")
                ax[i_index][j_index].set_ylabel("Reward")
                #         ax[i_index][j_index].set_title('Delay ' + str(delays[i_index]) + ', Sequence Length ' + str(sequence_lengths[j_index]))
                if len(dims_varied) > 1:
                    title_1st_dim = (
                        config_names[dims_varied[0]]
                        + " "
                        + str(dims_values[dims_varied[0]][i_index])
                    )
                    title_2nd_dim = (
                        config_names[dims_varied[1]]
                        + " "
                        + str(dims_values[dims_varied[1]][j_index])
                    )
                    ax[i_index][j_index].set_title(title_1st_dim + ", " + title_2nd_dim)
                else:
                    title_1st_dim = (
                        config_names[dims_varied[0]]
                        + " "
                        + str(dims_values[dims_varied[0]][j_index])
                    )
                    ax[i_index][j_index].set_title(title_1st_dim)
        #         ax[i_index][j_index].set_title('Sequence Length ' + str(seq_lens[j_index]))
        #         ax[i_index][j_index].set_title('Reward Density ' + str(reward_densities[j_index]))

        #         plt.legend(loc='upper left', prop={'size': 26})
        fig.tight_layout()
        # plt.suptitle("Training Learning Curves")
        plt.show()
        if save_fig:
            fig.savefig(
                stats_file.split("/")[-1]
                + ("_train" if train else "_eval")
                + "_learning_curves_"
                + str(exp_data["metric_names"][metric_num])
                + ".pdf",
                dpi=300,
                bbox_inches="tight",
            )  # Generates high quality vector graphic PDF 125kb; dpi doesn't matter for this

    ###### ---------- radar (spider) plot ------------- ######
    def plot_radar(
        self,
        list_exp_data,
        save_fig=False,
        train=True,
        metric_num=-2,
        plot_type="agent",
        weights={},
        use_aucs=False,
    ):
        """Plots radar (spider) chart across different dimensions

        Parameters
        ----------
        list_exp_data : list of experiment data of type <dict>
                   [ for more details refer load_data(), get_exp_data() ]
        save_fig : bool, optional
            A flag used to save a PDF (default is False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)
        metric_num :
            allowed values-> '-1' to plot episode mean lengths
                             '-2' to plot episode reward
        use_aucs : bool, optional
            A flag used to insert _aucs in the filename of the PDF (default is False)
        plot_type : string describing how to group data and plot, say based on agent or metric
            allowed values-> ['agent' ,'metric']
        weights : dictionary of weights associated per dimension (plot_type) data
            eg weights['reward_noise'] = [.25, .25, .25, .25]
        """

        stats_data = self.gather_stats(
            list_exp_data, train, metric_num, plot_type, use_aucs
        )

        # get axes labels
        first_group = next(iter(stats_data.values()))
        spoke_labels = first_group.keys()

        # plot
        N = len(spoke_labels)
        theta = radar_chart.radar_factory(N, frame="circle")
        plt.rcParams["figure.figsize"] = [7, 5]
        ax = plt.subplot(111, projection="radar")

        colors = [
            "blue",
            "orange",
            "green",
            "purple",
            "cyan",
            "olive",
            "brown",
            "grey",
            "red",
            "pink",
        ]
        for group_key, color in zip(stats_data.keys(), colors):
            d = []
            for sub_group_key in stats_data[group_key].keys():
                data = stats_data[group_key][sub_group_key]["to_plot_"]
                if sub_group_key in weights:
                    d.append(np.average(data, axis=0, weights=weights[sub_group_key]))
                else:
                    d.append(np.average(data))

            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)

        # customize plot axis labels
        ax.set_varlabels(spoke_labels)
        ax.set_rlabel_position(0)
        plt.xticks(color="black", fontsize=12)
        plt.yticks(color="black", fontsize=10)

        labels = stats_data.keys()
        legend = ax.legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize=12)

        # save figure
        if save_fig:
            fig_name = (
                stats_data[group_key][sub_group_key]["stats_file"].split("/")[-1]
                + ("_train" if train else "_eval")
                + ("_aucs" if use_aucs else "")
                + "_final_reward_"
                + stats_data[group_key][sub_group_key]["axis_labels"].replace(" ", "_")
                + "_"
                + str(stats_data[group_key][sub_group_key]["metric_names"][metric_num])
                + "_spider.pdf"
            )
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")

        plt.show()
