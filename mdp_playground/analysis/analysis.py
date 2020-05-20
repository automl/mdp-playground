import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class MDPP_Analysis():
    '''Utility class to load and plot data for analysis of experiments from MDP Playground

    '''
    def __init__(self):
        pass

    def load_data(self, dir_name, exp_name):
        '''Loads training and evaluation data from given file

        Parameters
        ----------
        dir_name : str
            The location where the training and evaluation CSV files were written
        exp_name : str
            The name of the experiment: the training and evaluation CSV filenames are formed using this string

        Returns
        -------
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
            print("Loading data from a sequential run of experiment configurations.")
        else:
            print("Loading data from a distributed run of experiment configurations. Creating a combined CSV stats file.")
            def join_files(file_prefix, file_suffix):
                '''Utility to join files that were written with different experiment configs'''
                with open(file_prefix + file_suffix, 'ab') as combined_file:
                    i = 0
                    while True:
                        if os.path.isfile(file_prefix + '_' + str(i) + file_suffix):
                            with open(file_prefix + '_' + str(i) + file_suffix, 'rb') as curr_file:
                                combined_file.write(curr_file.read())
                        else:
                            break
                        i += 1
            join_files(stats_file,  '.csv')
            join_files(stats_file, '_eval.csv')

        stats_pd = pd.read_csv(stats_file + '.csv', skip_blank_lines=True, header=None, comment='#', sep=' ')
        # print(stats_pd)
        # print(stats_pd[11].dtypes)
        # print(stats_pd.dtypes)
        # print(stats_pd.shape[0])

        final_rows_for_a_config = []
        previous_i = 0
        list_of_learning_curves = []
        # cols_to_take = 8

        for i in range(stats_pd.shape[0] - 1):
            if stats_pd.iloc[i, -3] > stats_pd.iloc[i + 1, -3]: #hardcoded: 3rd last column is no. of timesteps for the current run
                # list_of_learning_curves.append(stats_pd.iloc[previous_i:i+1, -cols_to_take:])
                previous_i = i + 1
                final_rows_for_a_config.append(i)
        # print("i, previous_i:", i, previous_i)
        final_rows_for_a_config.append(i + 1) # Always append the last row!
        # list_of_learning_curves.append(stats_pd.iloc[previous_i:i + 2, -cols_to_take:])
        self.final_rows_for_a_config = final_rows_for_a_config

        with open(stats_file + '.csv') as file_:
            config_names = file_.readline().strip().split(', ')
            config_names[0] = config_names[0][2:] # to remove '# ' that was written
            # config_names[-1] = config_names[-1][:-1] # to remove ',' that was written
        # print("config_names:", config_names)
        self.config_names = config_names # ['Delay', 'Sequence Length', 'Reward Density', 'Terminal State Density', 'P Noise', 'R Noise', 'dummy_seed']
        config_counts = []
        dims_values = []
        # For the following seeds should always be last column read! 1st column should be >=1 (it should not be 0 because that is the training_iteration that was recorded and is not used here)
        for i in range(1, 1 + len(config_names)): #hardcoded corresponds to columns written to evaluation stats CSV file
            dims_values.append(stats_pd[i].unique())
            config_counts.append(stats_pd[i].nunique())

        config_counts.append(3) #hardcoded number of training stats that were recorded
        config_counts = tuple(config_counts)

        # print(len(list_of_learning_curves))
        # print(len(final_rows_for_a_config))
        stats_end_of_training = stats_pd.iloc[final_rows_for_a_config]
        stats_reshaped = stats_end_of_training.iloc[:, -3:] #hardcoded # last vals are timesteps_total, episode_reward_mean, episode_len_mean
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
        hack_indices_10 = np.array(hack_indices) - 10
        # print(hack_indices_10.shape, hack_indices_10)
        # print(np.array(hack_indices[1:]) - np.array(hack_indices[:-1]))
        # print("Min:", min(np.array(hack_indices[1:]) - np.array(hack_indices[:-1]))) # Some problem with Ray? Sometimes no. of eval episodes is less than 10.
        final_10_evals = []
        for i in range(len(hack_indices)):
            final_10_evals.append(eval_stats[hack_indices_10[i]:hack_indices[i]])
        #     print(final_10_evals[-1])

        final_10_evals = np.array(final_10_evals) # has 2 columns: episode reward and episode length
        # print(final_10_evals.shape, final_10_evals)


        # final_vals = fin[final_rows_for_a_config]
        # print('final_rows_for_a_config', final_rows_for_a_config)
        # print("len(final_10_evals)", final_10_evals.shape, type(final_10_evals))
        mean_data_eval = np.mean(final_10_evals, axis=1) # this is mean over last 10 eval episodes
#         print(np.array(stats_pd.iloc[:, -3]))
        # Adds timesteps_total to the eval stats which did not have them:
        mean_data_eval = np.concatenate((np.atleast_2d(np.array(stats_pd.iloc[:, -3])).T, mean_data_eval), axis=1)
#         print(mean_data_eval.shape, len(final_rows_for_a_config))


        final_eval_metrics_ = mean_data_eval[final_rows_for_a_config, :] # 1st column is episode reward, 2nd is episode length
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
        for i in range(len(self.config_counts) - 1): # -1 is added to ignore seeds as dimensions of difficulty
            if self.config_counts[i]> 1:
                x_axis_labels.append(self.config_names[i])
                x_tick_labels_.append([str(j) for j in self.dims_values[i]])
                dims_varied.append(i)

        self.axis_labels = x_axis_labels
        self.tick_labels = x_tick_labels_
        self.dims_varied = dims_varied
        # print(x_axis_labels, x_tick_labels_, dims_varied)

        return stats_reshaped, final_eval_metrics_reshaped, np.array(stats_pd), mean_data_eval


    def plot_1d_dimensions(self, stats_data, save_fig=False, train=True):
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
        y_axis_label = 'Reward'

        plt.rcParams.update({'font.size': 18}) # default 12, for poster: 30
        # print(stats_data.shape)

        mean_data_ = np.mean(stats_data[..., -2], axis=-1)
        to_plot_ = np.squeeze(mean_data_)
        std_dev_ = np.std(stats_data[..., -2], axis=-1)
        to_plot_std_ = np.squeeze(std_dev_)

        plt.figure(figsize=(5, 1.5))

        print(to_plot_.shape)
        if len(to_plot_.shape) == 2: # Case when 2 meta-features were varied
            plt.bar(self.tick_labels[0], to_plot_[:, 0], yerr=to_plot_std_[:, 0])
        else:
            plt.bar(self.tick_labels[0], to_plot_, yerr=to_plot_std_)
        plt.xlabel(self.axis_labels[0])
        plt.ylabel(y_axis_label)
        if save_fig:
            plt.savefig(self.stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_final_reward_' + self.axis_labels[0].replace(' ','_') + '_1d.pdf', dpi=300, bbox_inches="tight")
        plt.show()

        if len(to_plot_.shape) == 2: # Case when 2 meta-features were varied
            plt.figure(figsize=(5, 1.5))
            plt.bar(self.tick_labels[1], to_plot_[0, :], yerr=to_plot_std_[0, :])
            # plt.tight_layout()
            plt.xlabel(self.axis_labels[1])
            plt.ylabel(y_axis_label)
            if save_fig:
                plt.savefig(self.stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_final_reward_' + self.axis_labels[1].replace(' ','_') + '_1d.pdf', dpi=300, bbox_inches="tight")
            plt.show()

    def plot_2d_heatmap(self, stats_data, save_fig=False, train=True):
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

        mean_data_ = np.mean(stats_data[..., -2], axis=-1)
        to_plot_ = np.squeeze(mean_data_)
        plt.imshow(np.atleast_2d(to_plot_), cmap='Purples', interpolation='none', vmin=0, vmax=np.max(to_plot_))
        plt.gca().set_xticklabels(self.tick_labels[1]) # dims 1 and 0 are exchanged here because Y-axis has plot for 1st varying dim and X-axis has plot for 2nd varying dim
        plt.gca().set_yticklabels(self.tick_labels[0])
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15 # default 15, for poster: 25
        cbar.set_label('Reward', rotation=270)
        plt.xlabel(self.axis_labels[1])
        plt.ylabel(self.axis_labels[0])
        if save_fig:
            plt.savefig(self.stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_final_reward_mean_heat_map.pdf', dpi=300, bbox_inches="tight")
        plt.show()
        std_dev_ = np.std(stats_data[..., -2], axis=-1)
        to_plot_ = np.squeeze(std_dev_)
        # print(to_plot_, to_plot_.shape)
        plt.imshow(np.atleast_2d(to_plot_), cmap='Purples', interpolation='none', vmin=0, vmax=np.max(to_plot_)) # 60 for DQN, 100 for A3C
        plt.gca().set_xticklabels(self.tick_labels[1])
        plt.gca().set_yticklabels(self.tick_labels[0])
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15 # default 15, for poster: 30
        cbar.set_label('Reward Std Dev.', rotation=270)
        plt.xlabel(self.axis_labels[1])
        plt.ylabel(self.axis_labels[0])
        # plt.tight_layout()
        if save_fig:
            plt.savefig(self.stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_final_reward_std_heat_map.pdf', dpi=300, bbox_inches="tight")
            # plt.savefig(stats_file.split('/')[-1] + '_train_heat_map.png')#, dpi=300)
        plt.show()

    def plot_learning_curves(self, stats_data, save_fig=False, train=True):
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
        # Plot for train metrics: learning curves; with subplot
        # Comment out unneeded labels in code lines 41-44 in this cell
        ncols_ = self.config_counts[self.dims_varied[0]]
        if len(self.dims_varied) > 1:
            nrows_ = self.config_counts[self.dims_varied[1]]
        else:
            nrows_ = 1
        nseeds_ = self.config_counts[-1]
        # print(ax, type(ax), type(ax[0]))
#         color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # print("color_cycle", color_cycle)
        plt.rcParams.update({'font.size': 25}) # 25 for 36x21 fig, 16 for 24x14 fig.
        # 36x21 for better resolution but about 900kb file size, 24x14 for okay resolution and 550kb file size
        fig, ax = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(7 * ncols_, 5 * nrows_))
        ax = np.atleast_2d(ax)
        # metrics_reshaped_squeezed = np.squeeze(metrics_reshaped)
        # print(np.squeeze(metrics_reshaped).shape)
        for i in range(len(self.final_rows_for_a_config)):
            i_index = i//(nseeds_ * ncols_) # = num_seeds * shape of more frequently changing hyperparam
            j_index = (i//nseeds_) % ncols_ #
            if i == 0:
                to_plot_ = stats_data[0:self.final_rows_for_a_config[i]+1,-2]
                to_plot_x = stats_data[0:self.final_rows_for_a_config[i]+1,-3]
            else:
                to_plot_ = stats_data[self.final_rows_for_a_config[i-1]+1:self.final_rows_for_a_config[i]+1, -2]
                to_plot_x = stats_data[self.final_rows_for_a_config[i-1]+1:self.final_rows_for_a_config[i]+1, -3]
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
                ax[i_index][j_index].set_title(self.config_names[self.dims_varied[0]] + ' ' + str(self.dims_values[self.dims_varied[0]][i_index]) + ', ' + self.config_names[self.dims_varied[1]] + ' '  + str(self.dims_values[self.dims_varied[1]][j_index]))
        #         ax[i_index][j_index].set_title('Sequence Length ' + str(seq_lens[j_index]))
        #         ax[i_index][j_index].set_title('Reward Density ' + str(reward_densities[j_index]))

        #         plt.legend(loc='upper left', prop={'size': 26})
        fig.tight_layout()
        # plt.suptitle("Training Learning Curves")
        plt.show()
        if save_fig:
            fig.savefig(self.stats_file.split('/')[-1] + ('_train' if train else '_eval') + '_learning_curves.pdf', dpi=300, bbox_inches="tight") # Generates high quality vector graphic PDF 125kb; dpi doesn't matter for this
