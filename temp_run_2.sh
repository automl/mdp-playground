#!/bin/bash

pwd
# date >>temp.txt
# # 103 Relaunch of 70; python custom_agents_noises.py 103 | tee 103.log
# python custom_agents.py 106 | tee 106.log # rerun of 26
# date >>temp.txt
# python custom_agents_rainbow_noises.py 110 | tee 110.log #
# date >>temp.txt


# python custom_agents_rainbow.py 123 | tee -a 123.log
# python custom_agents_rainbow1.py 123 | tee -a 123.log
# python custom_agents_rainbow2.py 123 | tee -a 123.log
# python custom_agents_rainbow3.py 123 | tee -a 123.log
# python custom_agents_rainbow4.py 123 | tee -a 123.log


# python custom_agents_rainbow_noises.py 122 | tee -a 122.log
# python custom_agents_rainbow_noises1.py 122 | tee -a 122.log
# python custom_agents_rainbow_noises2.py 122 | tee -a 122.log
# python custom_agents_rainbow_noises3.py 122 | tee -a 122.log
# python custom_agents_rainbow_noises4.py 122 | tee -a 122.log



# python custom_agents_a3c_noises.py 115 | tee -a 115.log
# python custom_agents_a3c_lstm_noises.py 116 | tee -a 116.log
#
# python custom_agents_sparsity.py 117 | tee -a 117.log
# python custom_agents_a3c_sparsity.py 118 | tee -a 118.log
# python custom_agents_a3c_lstm_sparsity.py 119 | tee -a 119.log
# python custom_agents_rainbow_sparsity.py 120 | tee -a 120.log

python custom_agents_sparsity_2.py 124 | tee 124.log
python custom_agents_a3c_sparsity_2.py 125 | tee 125.log
# python custom_agents_a3c_lstm_sparsity_2.py 126 | tee 126.log
# python custom_agents_rainbow_sparsity_2.py 127 | tee 127.log
