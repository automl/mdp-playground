import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import warnings

class MDPP_Analysis():
    '''Utility class to load and plot data for analysis of experiments from MDP Playground

    '''
    def __init__(self):
        pass
    
    def load_multiple_data(self, experiements: dict):
        '''Loads training and evaluation multiple data
        Parameters:
            key-value pair of exp_name & dir_name
            eg: "td3_move_to_a_point_action_max": "<path_to_data_dir>"
        for more details refer: load_data()
        '''
        exp_data = []
        for exp_name, dir_name in experiements.items():
            data = self.load_data(dir_name, exp_name)
            exp_data.append(data)
        return exp_data
    
    def load_data(self, dir_name, exp_name, num_metrics=3): #, max_total_configs=200):
        '''Loads training and evaluation data from given file

        Parameters
        ----------
        dir_name : str
            The location where the training and evaluation CSV files were written
        exp_name : str
            The name of the experiment: the training and evaluation CSV filenames are formed using this string

        Returns
        -------
        dictionary type with following data -
        train_stats : np.ndarray
            Training stats at end of training: 8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        eval_stats : np.ndarray
            Training stats at end of training: 8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        train_curves: np.ndarray
            The loaded training CSV with the last 3 columns the train stats that were saved and the initial columns are various setting for the algorithm and environment.
        eval_curves: np.ndarray
            The loaded evaluation CSV with the columns the evaluation stats that were saved
        '''

        stats_file = dir_name + '/' + exp_name #Name of file to which benchmark stats were written
        self.stats_file = stats_file

        if os.path.isfile(stats_file + '.csv'):
            print("Loading data from a sequential run/already combined runs of experiment configurations.")
        else:
            print("Loading data from a distributed run of experiment configurations. Creating a combined CSV stats file.")
            def join_files(file_prefix, file_suffix):
                '''Utility to join files that were written with different experiment configs'''
                with open(file_prefix + file_suffix, 'ab') as combined_file:
                    i = 0
                    missing_configs = []
                    num_diff_lines = []
                    while True: # i < max_total_configs:
                        if os.path.isfile(file_prefix + '_' + str(i) + file_suffix):
                            with open(file_prefix + '_' + str(i) + file_suffix, 'rb') as curr_file:
                                byte_string = curr_file.read()
                                newline_count = byte_string.count(10)
                                num_diff_lines.append(newline_count)
                                # if newline_count != 21 and file_suffix == '.csv': #hack to check only train files and not eval
                                #     warnings.warn('Expected 21 \\n chars in each stats file because we usually write stats every 1k timesteps for 20k timesteps. However, this can easily differ, e.g., for TD3 and DDPG where learning starts at 2k timesteps and there is 1 less \\n. Got only: ' + str(newline_count) + ' in file: ' + str(i))
                                combined_file.write(byte_string)
                        else:
                            # missing_configs.append(i)
                            break
                        i += 1
                    print(str(i) + " files were combined into 1 for file:" + file_prefix + '_n' + file_suffix)
                    # print("Files missing for config_nums:", missing_configs, ". Did you pass the right value for max_total_configs as an argument?")
                    # print("Unique line count values:", np.unique(num_diff_lines))
                    if i==0:
                        raise FileNotFoundError("No files to combine were present. Please check your location and/or filenames that they are correct.")
            join_files(stats_file,  '.csv')
            join_files(stats_file, '_eval.csv')

        stats_pd = pd.read_csv(stats_file + '.csv', skip_blank_lines=True, header=None, comment='#', sep=' ')
        # print(stats_pd)
        # print(stats_pd[11].dtypes)
        # print(stats_pd.dtypes)
        print("Training stats read (rows, columns):", stats_pd.shape)

        # Read column names
        with open(stats_file + '.csv') as file_:
            config_names = file_.readline().strip().split(', ')
            config_names[0] = config_names[0][2:] # to remove '# ' that was written
            # config_names[-1] = config_names[-1][:-1] # to remove ',' that was written
        # print("config_names:", config_names)
        self.config_names = config_names[1:] # ; begins at 1 to ignore training iteration num. ['Delay', 'Sequence Length', 'Reward Density', 'Terminal State Density', 'P Noise', 'R Noise', 'dummy_seed']
        config_counts = []
        dims_values = []
        # For the following seeds should always be last column read! 1st column should be >=1 (it should not be 0 because that is the training_iteration that was recorded and is not used here)
        for i in range(1, len(config_names) - num_metrics): # hardcoded 3 for no. of stats written
            dims_values.append(stats_pd[i].unique())
            config_counts.append(stats_pd[i].nunique())

        # config_counts[2] = 1 # hack
        # config_counts[-1] = 2 # hack for TD3 HP tuning files
        config_counts.append(num_metrics) #hardcoded number of training stats that were recorded
        config_counts = tuple(config_counts)
        self.metric_names = config_names[-num_metrics:]

        # Slice into training stats and get end of training stats for individual training runs in the experiment
        final_rows_for_a_config = []
        previous_i = 0
        list_of_learning_curves = []
        # cols_to_take = 8

        for i in range(stats_pd.shape[0] - 1):
            if stats_pd.iloc[i, -num_metrics] > stats_pd.iloc[i + 1, -num_metrics]: # hardcoded: -num_metrics column always HAS to be no. of timesteps for the current run
                # list_of_learning_curves.append(stats_pd.iloc[previous_i:i+1, -cols_to_take:])
                previous_i = i + 1
                final_rows_for_a_config.append(i)
        # print("i, previous_i:", i, previous_i)
        final_rows_for_a_config.append(i + 1) # Always append the last row!
        # list_of_learning_curves.append(stats_pd.iloc[previous_i:i + 2, -cols_to_take:])
        self.final_rows_for_a_config = final_rows_for_a_config

        # print(len(list_of_learning_curves))
        # print(len(final_rows_for_a_config))
        stats_end_of_training = stats_pd.iloc[final_rows_for_a_config]
        stats_reshaped = stats_end_of_training.iloc[:, -num_metrics:] # hardcoded # last vals are timesteps_total, episode_reward_mean, episode_len_mean
        stats_reshaped = np.reshape(np.array(stats_reshaped), config_counts)
        # print(stats_end_of_training.head(10))
        print("train stats shape:", stats_reshaped.shape)
#         to_plot_ = np.squeeze(stats_reshaped[:, :, :, :, 0, 0, :, 1])
#         print('Episode reward (at end of training) for 10 seeds for vanilla env.:', to_plot_)


        # Load evaluation stats
        stats_file_eval = stats_file + '_eval.csv'
        eval_stats = np.loadtxt(stats_file_eval, dtype=float)
        # print(eval_stats, eval_stats.shape)

        i = 0
        hack_indices = []
        for line in open(stats_file_eval):

            line=line.strip()
        #    print(line)
            if line.startswith("#HACK"):
        #         print(line, i)
                hack_indices.append(i - len(hack_indices)) # appends index of last eval in this training_iteration
            i += 1

        # print(len(hack_indices), hack_indices)
        if hack_indices[0] == 0: #hack
            hack_indices = hack_indices[1:] #hardcoded removes the 1st hack_index which is at position 0 so that hack_indices_10 below doesn't begin with a -10; apparently Ray seems to have changed logging for evaluation (using on_episode_end) from 0.7.3 to 0.9.0
            ray_0_9_0 = True
        else:
            ray_0_9_0 = False
        hack_indices_10 = np.array(hack_indices) - 10
        # print(hack_indices_10.shape, hack_indices_10)
        # print(np.array(hack_indices[1:]) - np.array(hack_indices[:-1]))
        # print("Min:", min(np.array(hack_indices[1:]) - np.array(hack_indices[:-1]))) # Some problem with Ray? Sometimes no. of eval episodes is less than 10.
        final_10_evals = []
        for i in range(len(hack_indices)):
            final_10_evals.append(eval_stats[hack_indices_10[i]:hack_indices[i]])
        #     print(final_10_evals[-1])
        if ray_0_9_0: #hack
            final_10_evals.append(eval_stats[hack_indices[i]:]) # appends the very last eval which begins at last hack_index for Ray 0.9.0

        final_10_evals = np.array(final_10_evals) # has 2 columns: episode reward and episode length
        # print(final_10_evals.shape, final_10_evals)


        # final_vals = fin[final_rows_for_a_config]
        # print('final_rows_for_a_config', final_rows_for_a_config)
        # print("len(final_10_evals)", final_10_evals.shape, type(final_10_evals))
        mean_data_eval = np.mean(final_10_evals, axis=1) # this is mean over last 10 eval episodes
#         print(np.array(stats_pd.iloc[:, -3]))

        # Adds timesteps_total column to the eval stats which did not have them:
        mean_data_eval = np.concatenate((np.atleast_2d(np.array(stats_pd.iloc[:, -num_metrics])).T, mean_data_eval), axis=1)
#         print(mean_data_eval.shape, len(final_rows_for_a_config))


        final_eval_metrics_ = mean_data_eval[final_rows_for_a_config, :] # 1st column is episode reward, 2nd is episode length in original _eval.csv file, here it's 2nd and 3rd after prepending timesteps_total column above.
        # print(dims_values, config_counts)
        final_eval_metrics_reshaped = np.reshape(final_eval_metrics_, config_counts)
        # print(final_eval_metrics_)
#         print("eval stats shapes (before and after reshape):", final_eval_metrics_.shape, final_eval_metrics_reshaped.shape)
        print("eval stats shape:", final_eval_metrics_reshaped.shape)

        self.config_counts = config_counts[:-1] # -1 is added to ignore "no. of stats that were saved" as dimensions of difficulty
        self.dims_values = dims_values

        # Catpure the dimensions that were varied, i.e. ones which had more than 1 value across experiments
        x_axis_labels = []
        x_tick_labels_ = []
        dims_varied = []
        for i in range(len(self.config_counts) - 1): # -1 is added to ignore #seeds as dimensions of difficulty
            if self.config_counts[i]> 1:
                x_axis_labels.append(self.config_names[i])
                x_tick_labels_.append([str(j) for j in self.dims_values[i]])
                for j in range(len(x_tick_labels_[-1])):
                    if len(x_tick_labels_[-1][j]) > 2: #hack
                        abridged_str = x_tick_labels_[-1][j].split(',')
                        if abridged_str[-1] == '':
                            abridged_str = abridged_str[:-1]
                        for k in range(len(abridged_str)):
                            if abridged_str[k] == 'scale':
                                abridged_str[k] = 'S'
                            elif abridged_str[k] == 'shift':
                                abridged_str[k] = 's'
                            elif abridged_str[k] == 'rotate':
                                abridged_str[k] = 'r'
                            elif abridged_str[k] == 'flip':
                                abridged_str[k] = 'f'
                            # abridged_str[j] = abridged_str[j][:2]
                        x_tick_labels_[-1][j] = ''.join(abridged_str)
                dims_varied.append(i)

        self.axis_labels = x_axis_labels
        self.tick_labels = x_tick_labels_
        self.dims_varied = dims_varied
        for d,v,i in zip(x_axis_labels, x_tick_labels_, dims_varied):
            print("Dimension varied:", d, ". The values it took:", v, ". Number of values it took:", config_counts[i], ". Index in loaded data:", i)

        
        data = dict()
        data['train_stats'] = stats_reshaped
        data['eval_stats'] = final_eval_metrics_reshaped
        data['train_curves'] = np.array(stats_pd)
        data['eval_curves'] = mean_data_eval
        
        data['metric_names'] = self.metric_names
        data['tick_labels'] = self.tick_labels[0]
        data['axis_labels'] = self.axis_labels[0]
        data['stats_file'] = self.stats_file
        data['algorithm'] = self.dims_values[0][0]
        data['dims_varied'] = self.dims_varied
        data['config_counts'] = self.config_counts
        data['final_rows_for_a_config'] = self.final_rows_for_a_config
        data['config_names'] = self.config_names
        data['dims_values'] = self.dims_values
        return data


    def plot_1d_dimensions(self, data, save_fig=False, train=True, metric_num=-2, plot_type = "agent"):
        exp_data = []
        if isinstance(data,  dict):
            #data from load_data
            exp_data.append(data)
        else:
            #data from load_multiple_data
            exp_data = data
            
        if isinstance(exp_data,  list):
            stats_data = dict()
            if plot_type is "agent":
                groupby = 'algorithm'
                legend = 'axis_labels'
            elif plot_type is "metric":
                groupby = 'axis_labels'
                legend = 'algorithm'
            
            
            for data in exp_data:
                if data[groupby][0] not in stats_data:
                    stats_data[data[groupby][0]] = dict()
                    stats_data[data[groupby][0]]['to_plot_'] = []
                    stats_data[data[groupby][0]]['to_plot_std_'] = []
                    stats_data[data[groupby][0]]['labels'] = []
                    
                if train:
                    stats = data['train_stats']
                else:
                    stats = data['eval_stats']
                
                
                mean_data_ = np.mean(stats[..., metric_num], axis=-1) # the slice sub-selects the metric written in position metric_num from the "last axis of diff. metrics that were written" and then the axis of #seeds becomes axis=-1 ( before slice it was -2).
                to_plot_ = np.squeeze(mean_data_)
                stats_data[data[groupby][0]]['to_plot_'].append(to_plot_)

                std_dev_ = np.std(stats[..., metric_num], axis=-1) #seed
                to_plot_std_ = np.squeeze(std_dev_)
                
                stats_data[data[groupby][0]]['to_plot_std_'].append(to_plot_std_)
                
                stats_data[data[groupby][0]]['labels'].append(data[legend])
                
                stats_data[data[groupby][0]]['tick_labels'] = data['tick_labels']
                stats_data[data[groupby][0]]['axis_labels'] = data[groupby]
                stats_data[data[groupby][0]]['metric_names'] = data['metric_names']
            
            show_legend=True
            
            #plot
            #plt.figure(1)
            rows = max(1, len(stats_data.keys())//2)
            cols = min(2, len(stats_data.keys())//2+1)
            figure, axes = plt.subplots(nrows=rows, ncols=cols) # [n * 2] shape
            i = 0
            for key in stats_data.keys():
                if rows == 1 and cols == 1:
                    self.plot_bar(axes, stats_data[key], save_fig, metric_num)
                    handles, labels = axes.get_legend_handles_labels()
                else:
                    self.plot_bar(axes[i], stats_data[key], save_fig, metric_num)
                    handles, labels = axes[i].get_legend_handles_labels()
                i += 1
            figure.legend(handles, labels, loc='upper center')
            figure.tight_layout()
            #plt.suptitle("Title centered above all subplots", fontsize=30)
            plt.show()            
        else:
            pass
        
    def plot_bar(self, ax, stats_data, save_fig=False, metric_num=-2):
        '''Plots 1-D bar plots across a single dimension with mean and std. dev.

        Parameters
        ----------
        stats_data : np.array
            8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        save_fig : bool, optional
            A flag used to save a PDF (default is
            False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)

        '''
        y_axis_label = 'Reward' if 'reward' in stats_data['metric_names'][metric_num] else stats_data['metric_names'][metric_num]

        to_plot_ = stats_data['to_plot_']
        to_plot_std_ = stats_data['to_plot_std_']
        labels = stats_data['labels']
        tick_labels =  stats_data['tick_labels']
        axis_labels =  stats_data['axis_labels']

        fig_width = len(tick_labels)
        plt.rcParams.update({'font.size': 18}) # default 12, for poster: 30
        plt.rcParams['figure.figsize'] = [fig_width*4, 5]
        
        width = 0.2
        x = np.arange(len(tick_labels))
        
        for idx in range(len(to_plot_)):
            ax.bar(x + width*idx, to_plot_[idx], width, yerr=to_plot_std_[idx], label=labels[idx])
        
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(axis_labels)
        #ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)

    def plot_2d_heatmap(self, data, save_fig=False, train=True, metric_num=-2):
        '''Plots 2 2-D heatmaps: 1 for mean and 1 for std. dev. across 2 meta-features of MDP Playground

        Parameters
        ----------
        stats_data : np.array
            8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        save_fig : bool, optional
            A flag used to save a PDF (default is
            False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)
        '''
        plt.rcParams.update({'font.size': 18}) # default 12, 24 for paper, for poster: 30
        cmap = 'Purples' # 'Blues' #
        label_ = 'Reward' if 'reward' in data['metric_names'][metric_num] else data['metric_names'][metric_num]
        
        tick_labels =  data['tick_labels']
        axis_labels =  data['axis_labels']
        stats_file = data['stats_file']
        
        if train:
            stats_data = data['train_stats']
        else:
            stats_data = data['eval_stats']

        mean_data_ = np.mean(stats_data[..., metric_num], axis=-1) #seed
        to_plot_ = np.squeeze(mean_data_)
        # print(to_plot_)
        if len(to_plot_.shape) > 2:
            # warning.warn("Data contains variation in more than 2 dimensions (apart from seeds). May lead to plotting error!")
            raise ValueError("Data contains variation in more than 2 dimensions (apart from seeds). This is currently not supported") #TODO Add 2-D plots for all combinations of 2 varying dims?
        plt.imshow(np.atleast_2d(to_plot_), cmap=cmap, interpolation='none', vmin=0, vmax=np.max(to_plot_))
        if len(tick_labels) == 2:
            plt.gca().set_xticklabels(tick_labels[1])
            plt.gca().set_yticklabels(tick_labels[0])
        else:
            plt.gca().set_xticklabels(tick_labels[0])
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15 # default 15, for poster: 25
        cbar.set_label(label_, rotation=270)
        if len(axis_labels) == 2:
            plt.xlabel(axis_labels[1])
            plt.ylabel(axis_labels[0])
        else:
            plt.xlabel(axis_labels[0])
        if save_fig:
            plt.savefig(stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_final_reward_mean_heat_map_' + str(data['metric_names'][metric_num]) + '.pdf', dpi=300, bbox_inches="tight")
        plt.show()
        std_dev_ = np.std(stats_data[..., metric_num], axis=-1) #seed
        to_plot_ = np.squeeze(std_dev_)
        # print(to_plot_, to_plot_.shape)
        plt.imshow(np.atleast_2d(to_plot_), cmap=cmap, interpolation='none', vmin=0, vmax=np.max(to_plot_)) # 60 for DQN, 100 for A3C
        if len(tick_labels) == 2:
            plt.gca().set_xticklabels(tick_labels[1])
            plt.gca().set_yticklabels([str(i) for i in tick_labels[0]])
        else:
            plt.gca().set_xticklabels(tick_labels[0])
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15 # default 15, for poster: 30
        cbar.set_label('Reward Std Dev.', rotation=270)
        if len(axis_labels) == 2:
            plt.xlabel(axis_labels[1])
            plt.ylabel(axis_labels[0])
        else:
            plt.xlabel(axis_labels[0])
        # plt.tight_layout()
        if save_fig:
            plt.savefig(stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_final_reward_std_heat_map_' + str(data['metric_names'][metric_num]) + '.pdf', dpi=300, bbox_inches="tight")
            # plt.savefig(stats_file.split('/')[-1] + '_train_heat_map.png')#, dpi=300)
        plt.show()

    def plot_learning_curves(self, data, save_fig=False, train=True, metric_num=-2): # metric_num needs to be minus indexed because stats_pd reutrned for train stats has _all_ columns
        '''Plots learning curves: Either across 1 or 2 meta-features of MDP Playground. Different colours represent learning curves for different seeds.

        Parameters
        ----------
        stats_data : np.array
            8-D tensor with 1st 6 dims the meta-features of MDP Playground, 7th dim is across the seeds, 8th dim is across different stats saved
        save_fig : bool, optional
            A flag used to save a PDF (default is
            False)
        train : bool, optional
            A flag used to insert either _train or _eval in the filename of the PDF (default is True)
        '''
        stats_file = data['stats_file']
        dims_varied = data['dims_varied']
        config_counts = data['config_counts']
        config_names = data['config_names']
        dims_values = data['dims_values']
        final_rows_for_a_config = data['final_rows_for_a_config']
        if train:
            stats_data = data['train_curves']
        else:
            stats_data = data['eval_curves']
        
        # Plot for train metrics: learning curves; with subplot
        # Comment out unneeded labels in code lines 41-44 in this cell
        if len(dims_varied) > 1:
            ncols_ = config_counts[dims_varied[1]]
            nrows_ = config_counts[dims_varied[0]]
        else:
            ncols_ = config_counts[dims_varied[0]]
            nrows_ = 1
        nseeds_ = config_counts[-1]
        # print(ax, type(ax), type(ax[0]))
#         color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # print("color_cycle", color_cycle)
        plt.rcParams.update({'font.size': 25}) # 25 for 36x21 fig, 16 for 24x14 fig.
        # 36x21 for better resolution but about 900kb file size, 24x14 for okay resolution and 550kb file size
        fig, ax = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(7 * ncols_, 5 * nrows_))
        ax = np.atleast_2d(ax)
        # metrics_reshaped_squeezed = np.squeeze(metrics_reshaped)
        # print(np.squeeze(metrics_reshaped).shape)
        for i in range(len(final_rows_for_a_config)):
            i_index = i//(nseeds_ * ncols_) # = num_seeds * shape of more frequently changing hyperparam
            j_index = (i//nseeds_) % ncols_ #
            if i == 0:
                to_plot_ = stats_data[0:final_rows_for_a_config[i]+1, metric_num]
                to_plot_x = stats_data[0:final_rows_for_a_config[i]+1,-3]
            else:
                to_plot_ = stats_data[final_rows_for_a_config[i-1]+1:final_rows_for_a_config[i]+1, metric_num]
                to_plot_x = stats_data[final_rows_for_a_config[i-1]+1:final_rows_for_a_config[i]+1, -3]
            # print(to_plot_[-1])
        #     if i % 10 == 0:
        #         fig = plt.figure(figsize=(12, 7))
        #     print(i//50, (i//10) % 5)
            ax[i_index][j_index].plot(to_plot_x, to_plot_, rasterized=False)#, label="Seq len" + str(seq_lens[i//10]))
            if i % nseeds_ == nseeds_ - 1: # 10 is num. of seeds
        #         pass
        #         print("Plot no.", i//10)
                ax[i_index][j_index].set_xlabel("Train Timesteps")
                ax[i_index][j_index].set_ylabel("Reward")
        #         ax[i_index][j_index].set_title('Delay ' + str(delays[i_index]) + ', Sequence Length ' + str(sequence_lengths[j_index]))
                if len(dims_varied) > 1:
                    title_1st_dim = config_names[dims_varied[0]] + ' ' + str(dims_values[dims_varied[0]][i_index])
                    title_2nd_dim = config_names[dims_varied[1]] + ' '  + str(dims_values[dims_varied[1]][j_index])
                else:
                    title_1st_dim = config_names[dims_varied[0]] + ' ' + str(dims_values[dims_varied[0]][j_index])
                    title_2nd_dim = ''
                ax[i_index][j_index].set_title(title_1st_dim + ', ' + title_2nd_dim)
        #         ax[i_index][j_index].set_title('Sequence Length ' + str(seq_lens[j_index]))
        #         ax[i_index][j_index].set_title('Reward Density ' + str(reward_densities[j_index]))

        #         plt.legend(loc='upper left', prop={'size': 26})
        fig.tight_layout()
        # plt.suptitle("Training Learning Curves")
        plt.show()
        if save_fig:
            fig.savefig(stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_learning_curves_' + str(data['metric_names'][metric_num]) + '.pdf', dpi=300, bbox_inches="tight") # Generates high quality vector graphic PDF 125kb; dpi doesn't matter for this
