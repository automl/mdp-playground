from cave.cavefacade import CAVE
from mdp_playground.analysis.mdpp_to_cave import MDPPToCave
import os

files_dir = "/mnt/484A4CAB4A4C9798/GoogleDrive/Maestria-Drive/HiWi/mdp_files"
experiments = {
#     "a3c_seq" : "%s/a3c_seq/"%files_dir,
#     "a3c_lstm_seq" : "%s/a3c_lstm_seq/"%files_dir,
#     "a3c_del" : "%s/a3c_del/"%files_dir,
    "a3c_lstm_del" : "%s/a3c_lstm_del/"%files_dir,
}

#The converted mdpp csvs will be stored in output_dir
output_dir = "../mdpp_to_cave"
mdpp_cave = MDPPToCave(output_dir)
for exp_name, v in experiments.items():
    dir_name, _ = os.path.split(v)
    cave_input_file = mdpp_cave.to_bohb_results(dir_name, exp_name)
    print(cave_input_file)
# Similarly, as an example, cave will ouput it's results 
# to the same directory as cave's input files

cave_results = os.path.join(cave_input_file, "out")
cave = CAVE(folders = [cave_input_file],
            output_dir = cave_results,
            ta_exec_dir = [cave_input_file],
            file_format = "BOHB",
            show_jupyter=True,
           )
cave.performance_table()
cave.local_parameter_importance()
# cave.cave_fanova()