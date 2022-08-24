# grep "Exit status: 1" mdp-playground-job-array.${JOB_ID}.* | cut -d'.' -f 3 2>&1 | tee mdpp_${JOB_ID}/failed_task_ids.txt

JOB_ID=$1
echo "Moving for JOB_ID: ${JOB_ID}" 
mkdir -p logs_mdpp_${JOB_ID}
mv mdp-playground-job-array.*${JOB_ID}* logs_mdpp_${JOB_ID}
# nemo specific: 
# mv mdpp.*${JOB_ID}* logs_mdpp_${JOB_ID}
# get failing task IDs based on cryptic Ray stderr message
# Meta specific 
#grep "actor died unexpectedly" logs_mdpp_${JOB_ID}/mdp-playground-job-array.${JOB_ID}* | cut -d'-' -f 2 | cut -d':' -f 1 2>&1 | tee mdpp_${JOB_ID}/failed_task_ids.txt
grep "Exit status: 1" logs_mdpp_${JOB_ID}/mdp-playground-job-array.${JOB_ID}* | cut -d'-' -f 2 | cut -d':' -f 1 2>&1 | tee mdpp_${JOB_ID}/failed_task_ids.txt
# Nemo specific
# grep "actor died unexpectedly" logs_mdpp_${JOB_ID}/mdpp.e${JOB_ID}* | cut -d'-' -f 2 | cut -d':' -f 1 2>&1 | tee mdpp_${JOB_ID}/failed_task_ids.txt

