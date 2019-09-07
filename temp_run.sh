#!/bin/bash

pwd
date >>temp.txt
# 103 Relaunch of 70
python custom_agents.py 106 | tee 106.log # rerun of 26
date >>temp.txt
python custom_agents_rainbow_noises.py 108 | tee 108.log # 
date >>temp.txt
