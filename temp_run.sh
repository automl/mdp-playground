#!/bin/bash

pwd
date >>temp.txt
# 103 Relaunch of 70; python custom_agents_noises.py 103 | tee 103.log
python custom_agents.py 106 | tee 106.log # rerun of 26
date >>temp.txt
python custom_agents_rainbow_noises.py 110 | tee 110.log # 
date >>temp.txt
