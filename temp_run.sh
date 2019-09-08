#!/bin/bash

pwd
# date >>temp.txt
# # 103 Relaunch of 70; python custom_agents_noises.py 103 | tee 103.log
# python custom_agents.py 106 | tee 106.log # rerun of 26
# date >>temp.txt
# python custom_agents_rainbow_noises.py 110 | tee 110.log #
# date >>temp.txt


python custom_agents_rainbow.py 121 | tee -a 121.log
python custom_agents_rainbow1.py 121 | tee -a 121.log
python custom_agents_rainbow2.py 121 | tee -a 121.log
python custom_agents_rainbow3.py 121 | tee -a 121.log
python custom_agents_rainbow4.py 121 | tee -a 121.log



# python custom_agents_a3c_noises.py 115 | tee -a 115.log
# python custom_agents_a3c_lstm_noises.py 116 | tee -a 116.log
#
# python custom_agents_sparsity.py 117 | tee -a 117.log
# python custom_agents_a3c_sparsity.py 118 | tee -a 118.log
# python custom_agents_a3c_lstm_sparsity.py 119 | tee -a 119.log
# python custom_agents_rainbow_sparsity.py 120 | tee -a 120.log
