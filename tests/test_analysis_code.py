import sys
from datetime import datetime
import logging
import copy
import numpy as np
from mdp_playground.envs.rl_toy_env import RLToyEnv
import unittest
import pytest


log_filename = (
    "/tmp/test_analysis_code_"
    + datetime.today().strftime("%m.%d.%Y_%I:%M:%S_%f")
    + ".log"
)  # TODO Make a directoy 'log/' and store there.


class TestAnalysisCode(unittest.TestCase):
    @pytest.mark.skip(
        reason="CAVE dependencies throw ImportError: cannot import name 'StatusType'"
    )
    def test_mdpp_to_cave(self):
        """ """
        print("\033[32;1;4mTEST_MDPP_TO_CAVE\033[0m")

        from mdp_playground.analysis import MDPP_Analysis

        # Set dir_name to the location where the CSV files from running an experiment were saved
        dir_name = "tests/files/mdpp_12744267_SAC_target_radius/"
        # Set exp_name to the name that was given to the experiment when running it
        exp_name = "sac_move_to_a_point_target_radius"
        # Set the following to True to save PDFs of plots that you generate below
        save_fig = True

        from cave.cavefacade import CAVE
        from mdp_playground.analysis.mdpp_to_cave import MDPPToCave
        import os

        # The converted mdpp csvs will be stored in output_dir
        output_dir = "/tmp/mdpp_to_cave"
        mdpp_cave = MDPPToCave()

        cave = mdpp_cave.to_CAVE_object(dir_name, exp_name, output_dir, overwrite=True)

        configs_json_line_1 = (
            '[[0, 0, 0], {"target_radius": 0.05}, {"model_based_pick": false}]'
        )

        with open(output_dir + "/" + exp_name + "/configs.json") as fh:
            l1 = fh.readline()[:-1]  # Ignore the \n at the end
            assert l1 == configs_json_line_1

        configspace_json_line_7 = '"default": 0.525,'

        with open(output_dir + "/" + exp_name + "/configspace.json") as fh:
            for i in range(7):
                l = fh.readline()
            assert l.strip() == configspace_json_line_7

        results_json_line_2 = '[[0, 0, 1], 20000, {"submitted": 1.1, "started": 1.2, "finished": 2.1}, {"loss": -7.261020890995861, "info": {}}, null]'

        with open(output_dir + "/" + exp_name + "/results.json") as fh:
            for i in range(2):
                l = fh.readline()
            assert l.strip() == results_json_line_2
