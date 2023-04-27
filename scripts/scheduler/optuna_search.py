#!/usr/bin/python
import sys
import os 
PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
sys.path.append(PATH_TO_LIB)
import itertools
import time
import shutil
import shlex
import json 
import hashlib
import stat
import subprocess
import time
import pickle
import random
import concurrent.futures
import optuna
from optuna.trial import TrialState
from threading import Thread
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from unplanned_net.utilities.collections import MaxTrialsCallback
import unplanned_net.utilities.parser as parser
from unplanned_net.utilities.database import MyDB

CMD = '''#!/bin/bash
'''
path = os.getcwd()
main_file = "main.py"

#here the additional parameter that are not included in the grid search are specified
args = parser.parse_args()

shutil.copytree(os.path.join(args.scripts_dir, "unplanned_net"), 'unplanned_snapshot', dirs_exist_ok=True)

moab_params = [k for argv in sys.argv[1:] for k in args.__dict__.keys() if k in argv]
additional_dict = {str(k) : args.__dict__[k] for k in moab_params}

print ("Runnign optuna from this study_id:\t{}\n".format(args.study_id))
try:
    config_file = json.load(open(os.path.join('./configs/', args.config_file)))
except NotADirectoryError:
    config_file = json.load(open('configs'))
param_dict = {}

if args.interactive:
   param_dict.update(config_file['static'])
else:
   param_dict.update(additional_dict)

def launch_experiment(filename, exp_id):
    try:
        subprocess.call("./moab_jobs/{} > logs/{}.log 2>&1".format(filename, filename[:-4]), shell=True)
        print("Launched exp: {}\n".format(filename))

        time.sleep(5)

        if args.optuna_metric == 'loss':
            result_table = pd.read_csv('CV_history_gridsearch.tsv', sep='\t')
            print ("Reading CV table for experiment id {} .. \n".format(exp_id))
            subset = result_table[result_table.exp_id == exp_id]
            optimization_param = np.amin(subset.val_loss.values)
            failure_value = np.inf 
        
        elif args.optuna_metric == 'mcc':
            result_table = pd.read_csv('AUC_history_gridsearch.tsv', sep='\t')
            print ("Reading AUC table for experiment id {} .. \n".format(exp_id))
            subset = result_table[result_table.exp_id == exp_id]
            optimization_param = np.amax(subset.val_mcc.values)
            failure_value = 0
        
        elif args.optuna_metric == 'auprc':
            result_table = pd.read_csv('AUC_history_gridsearch.tsv', sep='\t')
            print ("Reading AUC table for experiment id {} .. \n".format(exp_id))
            subset = result_table[result_table.exp_id == exp_id]
            optimization_param = np.amax(subset.val_auprc.values)
            failure_value = 0
        else:
            raise Exception("No action optuna when tuning metric is :\t{}".format(args.optuna_metric))
    
    except Exception as e:
        optimization_param = failure_value
        print ("\nExp {} failed with error {}.".format(exp_id, e))

    return optimization_param


def optim_metric(trial):
    #generate flag subsettring for the args your are tuning
    dynamic_args = {}
    for hp_type in config_file['dynamic']:
        for param in config_file['dynamic'][hp_type]:
            value = config_file['dynamic'][hp_type][param]
            if hp_type == 'suggest_categorical':
                dynamic_args[param] = getattr(trial, hp_type)(param, value)
            else:
                dynamic_args[param] = getattr(trial, hp_type)(param, *value)
 
    flag_dict = {**single_param_dict, **dynamic_args}
    param_string = '.'.join(['{}-{}'.format(k, flag_dict[k]) for k in sorted(list(flag_dict.keys()))])
    exp_id = hashlib.md5(param_string.encode()).hexdigest()
    flag_dict['exp_id'] = exp_id
    trial.set_user_attr('exp_id', exp_id)

    flag_string =  ''
    for k,v in flag_dict.items():
        flag_string += '--{} {} '.format(str(k),str(v))

    filename_list = ['study_id-{}'.format(args.study_id)]

    filename_list.append(f"exp_id-{flag_dict['exp_id']}")
    filename_list.append("exp")  # add file extension
    filename = '.'.join(filename_list)
    shell_arg = "python -u {} {}".format(main_file, flag_string)
    
    print ("Launching exp:\t{}".format(shell_arg))
    shellfile = open('moab_jobs/{}'.format(filename), 'w')
    shellfile.write(CMD)
    shellfile.write('cd ' + path + '\n')
    shellfile.write(shell_arg + '\n') #+ ' >logs/{}.log 2>&1 &\n'.format(file_name_joined))
    shellfile.close()
    st = os.stat('moab_jobs/{}'.format(filename))
    os.chmod('moab_jobs/{}'.format(filename), st.st_mode | stat.S_IEXEC)

    optimization_param = launch_experiment(filename, exp_id)
    return optimization_param

if args.interactive:
    iterable_param_dict = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))
    num_optuna_studies = len(list(itertools.product(*param_dict.values())))
    print ("Running interactively a sequence of {} hp_searces..\n".format(num_optuna_studies))
else:
    iterable_param_dict = [param_dict] 
    num_optuna_studies =1

for it, single_param_dict in enumerate(iterable_param_dict):    

    if "optuna_iters" in single_param_dict:
        max_evals = single_param_dict["optuna_iters"]
    else:
        max_evals = args.optuna_iters

    if args.interactive:
        args.study_id = hashlib.md5('.'.join(["{}-{}".format(k,single_param_dict[k]) for k in sorted(single_param_dict)]).encode()).hexdigest()
    
    print ("Starting optuna study {} with {} optuna exp. {}/{}\n".format(args.study_id, max_evals, it+1, num_optuna_studies))

    MyDB().query("CREATE SCHEMA IF NOT EXISTS {};".format(args.optuna_schema))

    storage = optuna.storages.RDBStorage(
                            url="postgresql://daplaci@trans-db-01/daplaci?options=-c%20search_path={}".format(args.optuna_schema), 
                            engine_kwargs={"pool_size":0})

    metric_direction = "minimize" if args.optuna_metric == "loss" else "maximize"
    
    if 'grid' in config_file:
        sampler = optuna.samplers.GridSampler(config_file['grid'])
    else:
        sampler = optuna.samplers.TPESampler(multivariate=True)

    study = optuna.create_study(
                        storage=storage, study_name=args.study_id, load_if_exists=True, direction=metric_direction,
                        pruner=optuna.pruners.HyperbandPruner(
                        min_resource=15, reduction_factor=3),
                        sampler=sampler)

    for k,v in single_param_dict.items():
        #save in the study some information about the statis parameters used
        study.set_user_attr(k, v)
    
    study.optimize(optim_metric, callbacks=[MaxTrialsCallback(max_evals, states=(TrialState.COMPLETE,))])

Path("logs/{}_completed".format(args.study_id)).touch()
sys.exit(0)