JOB_ID=$1
echo "Moving for JOB_ID: ${JOB_ID}" 
mkdir -p logs_mdpp_${JOB_ID}
mv mdpp.*${JOB_ID}* logs_mdpp_${JOB_ID}
# get failing task IDs based on cryptic Ray stderr message
grep "actor died unexpectedly" logs_mdpp_${JOB_ID}/mdpp.e${JOB_ID}* | cut -d'-' -f 2 | cut -d':' -f 1 2>&1 | tee mdpp_${JOB_ID}/failed_task_ids.txt

