# Examples: 
# py generate_mdpp_plots.py -f expt_list.txt
# py generate_mdpp_plots.py --exp-id 13699485 --exp-name dqn_del # --show-plots  # exp_id and exp_name
# Setup to analyse an MDP Playground experiment
from mdp_playground.analysis import MDPP_Analysis

import yaml
import argparse

from collections import Counter

# Based on https://stackoverflow.com/a/71751051/11063709, to allow keys to have a list of values
# in case duplicate keys are present in the YAML.
def parse_preserving_duplicates(src):
    class PreserveDuplicatesLoader(yaml.loader.Loader):
        pass

    def map_constructor(loader, node, deep=False):
        keys = [loader.construct_object(node, deep=deep) for node, _ in node.value]
        vals = [loader.construct_object(node, deep=deep) for _, node in node.value]
        key_count = Counter(keys)
        data = {}
        for key, val in zip(keys, vals):
            if key_count[key] > 1:
                if key not in data:
                    data[key] = []
                data[key].append(val)
            else:
                data[key] = [val]
        return data

    PreserveDuplicatesLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, map_constructor)
    return yaml.load(src, PreserveDuplicatesLoader)


def generate_plots(exp_name, exp_id, show_plots=False, options=''):
    print("Generating plots for " + str(exp_id) + ": " + exp_name + " with the following addnl. options: " + options)
    
    # Set dir_name to the location where the CSV files from running an experiment were saved
    dir_name = '/home/rajanr/mdpp_' + str(exp_id) # e.g. 13699485
    # Set exp_name to the name that was given to the experiment when running it
    # exp_name = 'dqn_del'
    # Set the following to True to show plots that you generate below
    # show_plots = True
    # Set the following to True to save PDFs of plots that you generate below
    save_fig = True
    err_bar = 'bootstrap'  # 't_dist'
    bonferroni = True
    if 'eval' in options:
        load_eval = True
    else:
        load_eval = False
    if 'auto_y_scale' in options:
        common_y_scale = False
    else:
        common_y_scale = True

    # Data loading
    mdpp_analysis = MDPP_Analysis()
    train_stats, eval_stats, train_curves, eval_curves, train_aucs, eval_aucs = mdpp_analysis.load_data(dir_name, exp_name, load_eval=load_eval)

    # 1-D: Plots showing reward after total timesteps when varying a single meta-feature
    # Plots across n runs: Training: with std dev across the runs
    mdpp_analysis.plot_1d_dimensions(train_aucs, save_fig, bonferroni=bonferroni, err_bar=err_bar, show_plots=show_plots, common_y_scale=common_y_scale)

    if 'ep_len' in options:
        mdpp_analysis.plot_1d_dimensions(train_aucs, save_fig, bonferroni=bonferroni, err_bar=err_bar, show_plots=show_plots, metric_num=-1)

    # 2-D heatmap plots across n runs: Training runs: with std dev across the runs
    # There seems to be a bug with matplotlib - x and y axes tick labels are not correctly set even though we pass them. Please feel free to look into the code and suggest a correction if you find it.
    if 'plot_2d' in options:
        mdpp_analysis.plot_2d_heatmap(train_aucs, save_fig, show_plots=show_plots, common_y_scale=common_y_scale)

        if 'ep_len' in options:
            mdpp_analysis.plot_2d_heatmap(train_aucs, save_fig, show_plots=show_plots, common_y_scale=common_y_scale, metric_num=-1)

    # Plot learning curves: Training: Each curve corresponds to a different seed for the agent
    if 'learn_curves' in options:
        mdpp_analysis.plot_learning_curves(train_curves, save_fig, show_plots=show_plots, common_y_scale=common_y_scale)

    if 'eval' in options:
        mdpp_analysis.plot_1d_dimensions(eval_aucs, save_fig, bonferroni=bonferroni, err_bar=err_bar, show_plots=show_plots, common_y_scale=common_y_scale, train=False)

        if 'ep_len' in options:
            mdpp_analysis.plot_1d_dimensions(eval_aucs, save_fig, bonferroni=bonferroni, err_bar=err_bar, show_plots=show_plots, metric_num=-1, train=False)

        if 'plot_2d' in options:
            mdpp_analysis.plot_2d_heatmap(eval_aucs, save_fig, show_plots=show_plots, common_y_scale=common_y_scale, train=False)

            if 'ep_len' in options:
                mdpp_analysis.plot_2d_heatmap(eval_aucs, save_fig, show_plots=show_plots, common_y_scale=common_y_scale, metric_num=-1, train=False)

        # Plot learning curves: Training: Each curve corresponds to a different seed for the agent
        if 'learn_curves' in options:
            mdpp_analysis.plot_learning_curves(eval_curves, save_fig, show_plots=show_plots, common_y_scale=common_y_scale, train=False)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Process Latex .bib files")

    parser.add_argument(
        "--exp-file", "-f", type=str, help="Expt. identifiers and names listed in a YAML file, i.e., JOB_ID: exp_name",
    )

    parser.add_argument(
        "--exp-id", "-i", type=str, help="Expt. identifier, i.e., JOB_ID from cluster"
    )

    parser.add_argument(
        "--exp-name", "-e", type=str, help="expt name, corresponds to the names of the CSV stats files and the <config>.py file used for the expt."
    )

    parser.add_argument(
        "--show-plots", "-p", action='store_true', dest='show_plots', help="Toggle displaying plots", default=False,
    )

    parser.add_argument(
        "--num-expts", "-n", type=int, help="First n expts in the list are plotted"
    )

    args = parser.parse_args()

    # print(args)

    if args.exp_file is not None:
        with open(args.exp_file) as f:
            yaml_dict = parse_preserving_duplicates(f)  # yaml.safe_load(f)

        print("List of expts.:", yaml_dict)

        i = 0
        for exp_id in yaml_dict:
            if len(yaml_dict[exp_id]) > 1:
                print("More than 1 expt. for the same expt_id:", exp_id, ". The expts.:", yaml_dict[exp_id])
            for j in range(len(yaml_dict[exp_id])):
                i += 1
                print("\nExpt. no.:", i , "from the list.")

                exp_name = yaml_dict[exp_id][j].split(' ')[0]
                options = ' '.join(yaml_dict[exp_id][j].split(' ')[1:]) if ' ' in yaml_dict[exp_id][j] else ''
                # if 'learn_curves' in options:
                generate_plots(exp_id=exp_id, exp_name=exp_name, show_plots=args.show_plots, options=options)

                # Need to break out of 2 for loops
                if args.num_expts is not None and i == args.num_expts:
                    break

            if args.num_expts is not None and i == args.num_expts:
                break


    else:
        dict_args = vars(args)
        del dict_args['exp_file']
        dict_args['exp_id'] = dict_args['exp_id'].split(' ')[0]
        dict_args['options'] = ' '.join(dict_args['exp_id'].split(' ')[1:]) if ' ' in dict_args['exp_id'] else ''
        # print(dict_args)
        generate_plots(**dict_args)


