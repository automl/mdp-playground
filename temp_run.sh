#!/bin/bash

pwd
# date >>temp.txt
# # 103 Relaunch of 70; python custom_agents_noises.py 103 | tee 103.log
# python custom_agents.py 106 | tee 106.log # rerun of 26
# date >>temp.txt
# python custom_agents_rainbow_noises.py 110 | tee 110.log #
# date >>temp.txt


python custom_agents_rainbow_noises.py 113 | tee -a 113.log
python custom_agents_rainbow_noises1.py 113 | tee -a 113.log
python custom_agents_rainbow_noises2.py 113 | tee -a 113.log
python custom_agents_rainbow_noises3.py 113 | tee -a 113.log
python custom_agents_rainbow_noises4.py 113 | tee -a 113.log
