#!/usr/bin/python
import sys
import os
from os.path import dirname, realpath
sys.path.append((dirname(dirname(realpath(__file__)))))
import itertools
import time
import shutil
import stat
import subprocess
import json
import hashlib
import unplanned_net.utilities.parser as parser

CMD = """#!/bin/bash
#PBS -l nodes=1:ppn={}{}
#PBS -l mem={}gb
#PBS -l walltime=5:00:00:00
#PBS -N lpr_2018
#PBS -e moab_jobs/$PBS_JOBID.err
#PBS -o moab_jobs/$PBS_JOBID.out

module load singularity/3.7.3
module load anaconda3/2021.05

# Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
export LC_CTYPE=en_US.UTF-8

SCRATCH_DIR="/local/scratch/$PBS_JOBID"
export SINGULARITY_TMPDIR="$SCRATCH_DIR/singularity/tmp"
export SINGULARITY_CACHEDIR="$SCRATCH_DIR/singularity/cache"
IMGDIR="$SCRATCH_DIR/singularity/img_dir"

mkdir -p $SCRATCH_DIR
mkdir -p $SINGULARITY_TMPDIR
mkdir -p $SINGULARITY_CACHEDIR
mkdir -p $IMGDIR

SINGULARITY_NAME='unplanned_container.sif'
SINGULARITY_PATH="/users/singularity/$SINGULARITY_NAME"
cp $SINGULARITY_PATH $IMGDIR
cd {}
echo Singularity copied in $IMGDIR
singularity run --nv $PBS_O_WORKDIR:$PBS_O_WORKDIR $IMGDIR/$SINGULARITY_NAME {}"""

def write_and_launch(worker):

    all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))
    cores = param_dict["num_workers"][0]*2
    if "use_gpu" in param_dict and param_dict["use_gpu"][0]:
        use_gpu = ":gpus=1"
    else:
        use_gpu = ""

    for arg in all_args:
        shell_arg = 'python -u optuna_search.py '
        arg.update(additional_dict)
        arg.update({'date_string':date_string})
        for key in sorted(arg):
            shell_arg += ' --' + key + ' ' + str(arg[key])
        if 'study_id' not in arg:
            study_id = hashlib.md5('.'.join(["{}-{}".format(k,arg[k]) for k in sorted(arg)]).encode()).hexdigest()
            shell_arg += ' --study_id {}'.format(study_id)
        shell_arg += " > logs/{}_{}.log 2>&1".format(study_id, worker)
        shellfile = open('moab_jobs/{}_{}.sh'.format(study_id, worker), 'w')
        shellfile.write(CMD.format(cores, use_gpu, param_dict["mem"][0], path, shell_arg))
        shellfile.close()

#here the additional parameter that are not included in the grid search are specified
args = parser.parse_args()
moab_params = [k for argv in sys.argv[1:] for k in args.__dict__.keys() if k==argv[2:]]
additional_dict = {str(k) : args.__dict__[k] for k in moab_params}

print ("additional params:", additional_dict)
# check if Python-version > 3.0
# with Python2 the dict is not in alphabetic order
assert (sys.version_info > (3, 0)), "This script only works with Python3!"

script_dir = dirname(dirname(realpath(__file__)))
base_dir = dirname(script_dir)
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")
scheduler_dir = os.path.join(script_dir, "scheduler")
summarizer_dir = os.path.join(script_dir, "summarizer")
batch_file = 'run_models.sh'
list_file_to_copy = os.listdir(script_dir)


# make parameter dictionary
param_dict={}

try:
    config_file = json.load(open(os.path.join('./configs/', args.config_file)))
    print ("Running from config file: \t{}".format(args.config_file))
except:
    raise Exception("if you do NOT run interactively and you want to directly dispatch the jobs, you have to specify a config file")

param_dict.update(config_file['static'])

# do not run this, if an other script is just trying to import 'param_dict'
if __name__ == '__main__':

    unknown_param = [k for k in list(param_dict.keys()) if k not in args.__dict__.keys()]
    if any(unknown_param):
        raise Warning("These parameters of the grid search are not initialized in the parser: \n{}".format('\n'.join(unknown_param)))

    # make list of all argument combinations
    # all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values())
    
    # change directory and open files for writing
    date_string = time.strftime("%Y-%m-%d-%H%M")
    name_folder_add = input("Type the name of the output folder to append at the current date time. Insert a string without spaces: ")
    date_string += name_folder_add
    
    os.chdir(output_dir)
    
    if os.path.exists(name_folder_add):
        raise NotImplementedError

    os.mkdir(date_string)
    path = os.path.join(output_dir, date_string)
    print ("Run your grid_search in the folder called:", path)
    os.chdir(path)
    os.mkdir('best_weights')
    os.mkdir('moab_jobs')
    os.mkdir('data')
    os.mkdir('logs')
    os.mkdir('figures')
    
    # save this script to path
    filename = os.path.join(base_dir, __file__ )
    shutil.copy2(filename, path)
    for file_to_copy in list_file_to_copy:
        if ('.py' in file_to_copy) or ('.R' in file_to_copy):
            shutil.copy2(os.path.join(script_dir, file_to_copy), path)
    
    shutil.copy2(os.path.join(scheduler_dir, "optuna_search.py"), path)
    shutil.copy2(os.path.join(summarizer_dir, "collect_results.py"), path)
    
    if args.interactive:
        shutil.copytree(os.path.join(base_dir, 'configs'), os.path.join(path, 'configs'))
    else:
        shutil.copy2(os.path.join(base_dir, 'configs', args.config_file), os.path.join(path, 'configs'))


    infile = open(os.path.join(script_dir, 'summarizer/eval_gridsearch.R'), 'r') 
    rscript = infile.read()
    rscript = rscript.replace('#placeholder#', output_dir + date_string)
    infile.close()
    outfile = open('eval_gridsearch.R', 'w')
    outfile.write(rscript)
    outfile.close()
      
    # make batch-file
    batch_file = open('run_models.sh', 'w')
    batch_file.write('#!/bin/bash\n')
    batch_file.write("""#PBS -l nodes=1:ppn=1
#PBS -l mem=1gb
#PBS -l walltime=20:00:00:00
#PBS -e logs/$PBS_JOBID.err
#PBS -o logs/$PBS_JOBID.log
#PBS -N dispatcher
""")
    batch_file.write('cd ' + output_dir + date_string + '/\n')
    batch_file.write("""
for f in moab_jobs/*.sh
do
    study_id=$(echo $f | cut -d '_' -f 1)
    study_id=${study_id}_completed
    if [[ -f logs/$study_id ]]
    then
        continue
    fi
    num_jobs=$(showq -u $(whoami) | grep $(whoami) | wc -l)
    echo "$num_jobs"
    while [ $num_jobs -gt 40 ]
    do
        echo "Waiting for less job in queue"
        sleep 10
        num_jobs=$(showq -u $(whoami) | grep $(whoami) | wc -l)
    done
    if [ $(hostname) == 'precision05']
    then
        echo "sending job via ssh on p05"
        echo "cd ${PWD} && qsub $f" | ssh precision05 bash
    else
        echo "I am p05: qsub"
        qsub $f
    fi
    echo "running"
    sleep 5
done
echo "\n### Scheduler Launched all the study worker"
""") 


    batch_file.close()
    # change permissions
    st = os.stat('run_models.sh')
    os.chmod('run_models.sh', st.st_mode | stat.S_IEXEC)
    
    batch_file = open('collect_results.sh', 'w')
    cores = param_dict["num_workers"][0]*2
    if "use_gpu" in param_dict and param_dict["use_gpu"][0]:
        use_gpu = ":gpus=1"
    else:
        use_gpu = ""
    batch_file.write(CMD.format(cores, use_gpu, param_dict["mem"][0], path, 
    "python -u collect_results.py --apply_calibration --n_samples 10 --plot_attribution > logs/collector.log"))
    batch_file.close()
    st = os.stat('collect_results.sh')
    os.chmod('collect_results.sh', st.st_mode | stat.S_IEXEC)

    # create output files
    AUCfile = open('AUC_history_gridsearch.tsv', 'w')
    CVfile = open('CV_history_gridsearch.tsv', 'w')
    
    static_keys = list(param_dict.keys())
    dynamic_keys = [subd for d in config_file["dynamic"] for subd in config_file["dynamic"][d]]
    
    AUCheader = ['mcc', 'val_mcc', 'precision', 'val_precision', 
                'recall', 'val_recall', 'auc','val_auc', 'auprc', 'val_auprc',
                'event', 'time', 'baseline_hours', 'time_to_event',
                'cv_num', 'model', 'exp_id', 'study_id'] + static_keys + dynamic_keys
    
    CVheader = ['auc', 'loss',  'mcc', 'precision', 'recall', 
                'val_auc', 'val_loss',   'val_mcc', 'val_precision',  'val_recall', 'auprc', 'val_auprc',
                'cv_num', 'event', 'time', 'time_to_event', 'exp_id'] + static_keys + dynamic_keys
                        
    CVfile.write('epoch\t' + '\t'.join(CVheader) + '\n')
    AUCfile.write('\t'.join(AUCheader) + '\n')

    # close files
    AUCfile.close()
    CVfile.close()
    
    # open log files
    error_log = open('error.log', 'w')
    errorHeader = ['error', 'args']
    error_log.write('\t'.join(errorHeader) + '\n')
    error_log.close()
    
    progress_log = open('progress.log', 'w')
    progressHeader = ['completed']
    progress_log.write('\t'.join(progressHeader) + '\n')
    progress_log.close()

    for worker in range(param_dict["num_workers_per_study"][0]):
        write_and_launch(worker)
        
    if not args.interactive:
        #run batch-scripts
        run_models_path = '/'.join([path, r'run_models.sh'])
        subprocess.call("nohup {} > logs/scheduler_{}.log &".format(run_models_path, date_string), shell=True)
  
