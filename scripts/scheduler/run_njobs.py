import sys,os, subprocess, time, random

jobid = [file for file in os.listdir("moab_jobs/")]

random.shuffle(jobid)

job_gen = (i for i in jobid)

while True:
    process_runnign = "main" in str(subprocess.check_output("ps ux", shell=True))
    if not process_runnign:
        current_job  = next(job_gen)
        print ("Launching experiment {}".format(current_job))
        assert os.access("./moab_jobs/{}".format(current_job), os.X_OK)
        subprocess.call("./moab_jobs/{} > logs/{}.log 2>&1".format(current_job, current_job[:-3]), shell=True)
        print ("Exp finished {}".format(current_job))
        os.system("rm ./moab_jobs/{}".format(current_job))
    else:
        print ("JOb is still running")
        time.sleep(60)
