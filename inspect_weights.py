import tensorflow as tf
import stable_baselines as sb
import matplotlib.pyplot as plt
from stable_baselines.common import BaseRLModel
from stable_baselines import DQN, DDPG, SAC, A2C, TD3
import numpy as np
import argparse
import zipfile
import os
import sys
import gym


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__) # docstring at beginning of the file is stored in __doc__
    parser.add_argument('-n', '--file-name', dest="filename",
                        help="full path where the stable baselines model is stored as a zip")
    parser.add_argument('-s', '--save-dir', dest="save_dir", default="../tmp/",
                        help="Where to save the images")
    args = parser.parse_args()
    print("Parsed args:", args)
    return args


def main(args):
    env = gym.make('CartPole-v1')
    model = A2C(sb.common.policies.MlpLstmPolicy, env)
    # model.load(args.filename)
    f_path = os.path.abspath(args.filename)
    if(os.path.exists(f_path+".zip")):
        _, params = model._load_from_file(f_path)
    else:
        print("Path does not exist: %s" % f_path)
        return

    if(not os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    w = tf.summary.FileWriter("../mdp_files/output/", model.graph)

    min_v, max_v = sys.maxsize, -sys.maxsize - 1
    for v in params.values():
        if(np.max(v) > max_v):
            max_v = np.max(v)
        elif(np.min(v) < min_v):
            min_v = np.min(v)

    exp_name = os.path.abspath(args.filename).split(os.path.sep)[-1]
    for k, v in params.items():
        if('w' in k):  # weights
            p_name = k.split('/')[1]
            plt.suptitle(p_name + str(v.shape))
            plt.imshow(v, cmap="twilight")
            # plt.show()
            plt.clim(min_v, max_v)
            plt.colorbar()
            save_plot = os.path.join(
                            args.save_dir,
                            "%s_%s" % (exp_name, p_name))
            plt.savefig(save_plot, bbox_inches='tight',
                        pad_inches=0.0, dpi=200)
            plt.clf()
            plt.close()
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
