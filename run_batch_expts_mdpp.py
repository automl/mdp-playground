#Rough and experimental

import sys
import os

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

                os.system("pwd")

                exp_name = yaml_dict[exp_id][j].split(' ')[0]

                cmd = "~/mdp-playground/tabular_rl/run_experiments_tabular.py --exp-name " + exp_name + " --config-file ~/mdp-playground/experiments/" + exp_name + ".py"

                options = ' '.join(yaml_dict[exp_id][j].split(' ')[1:]) if ' ' in yaml_dict[exp_id][j] else ''
                # if 'learn_curves' in options:
                os.system(sys.executable + " " + cmd) # sys.executable contains "current" Python
                # print(sys.executable + " " + cmd)

                cp_cmd = "cp " + exp_name + ".csv ~/mdpp_" + exp_name + "/"
                # print(cp_cmd)
                os.system(cp_cmd)

                # Need to break out of 2 for loops
                if args.num_expts is not None and i == args.num_expts:
                    break

            if args.num_expts is not None and i == args.num_expts:
                break


    else:
        print("Not implemented")
        sys.exit(1)
        dict_args = vars(args)
        del dict_args['exp_file']
        dict_args['exp_id'] = dict_args['exp_id'].split(' ')[0]
        dict_args['options'] = ' '.join(dict_args['exp_id'].split(' ')[1:]) if ' ' in dict_args['exp_id'] else ''
        # print(dict_args)

        # exp_name = yaml_dict[exp_id][j].split(' ')[0]

        # cmd = "~/mdp-playground/tabular_rl/run_experiments_tabular.py  --exp-name " + exp_name + "--config-file experiments/" + exp_name + ".py"

        # os.system(sys.executable + cmd) # sys.executable contains "current" Python

