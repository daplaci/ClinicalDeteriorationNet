import subprocess
import pprint
import argparse
import pandas as pd
import numpy as np
import os
from os.path import dirname, realpath
import traceback
import hashlib
from collections import defaultdict

def get_argument_parser(parent_parser=[]):
    def str2bool(v):
        if isinstance(v, bool):
           return v 
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return str(v)
    
    parser = argparse.ArgumentParser(description='Unplanned ICU model', parents=parent_parser)
    # What to execute
    #training params
    parser.add_argument('--cv_folds', type=int, default=3, help='number cross folds')
    parser.add_argument('--n_epochs', type=int, default=5000, help='number of epochs for train [default: 256]')
    parser.add_argument('--steps_per_epoch_train', type=int, default=10, help='num of batch to sample for each epoch in training')
    parser.add_argument('--steps_per_epoch_val', type=int, default=15, help='num of batch to sample for each epoch in validation')
    parser.add_argument('--patience', type=int, default=100, help='epochs to wait before stopping')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropuout to apply')
    parser.add_argument('--batch_norm', type=str2bool, default=False, help='flag to apply batch normalization')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--alpha_loss', type=float, default=1.0, help='dropuout to apply')
    parser.add_argument('--beta_loss', type=float, default=1.0, help='dropuout to apply')
    parser.add_argument('--layers', type=int, default=1, help='number of layers to use in mlp')
    parser.add_argument('--units', type=int, default=64, help='number of units to use')
    parser.add_argument('--embedding_dim', type=int, default=256, help='seed for random')
    parser.add_argument('--embedding_coeff', type=float, default=1, help='coefficient used for computing the emb size from the vocab size')
    parser.add_argument('--l2_regularizer', type=float, default=1e-4, help='l2 regularizer to apply in the model')

    parser.add_argument('--test_size', type=float, default=0.2, help='percentage of test to held out')
    parser.add_argument('--activation', type=str, default='', help='activation for net')
    parser.add_argument('--seed', type=int, default=42, help='seed for random')
    parser.add_argument('--train_batch', type=int, default=128, help='batch_size training')
    parser.add_argument('--val_batch', type=int, default=128, help='batch_size validation')
    parser.add_argument('--masking', type=str2bool, default=True, help='bool for masking')
    parser.add_argument('--early_stopping', type=str2bool, default=True, help='wheter to apply early stopping')
    parser.add_argument('--save_checkpoint', type=str2bool, default=True, help='wheter to save checkpoint')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='optmizer for masking')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers for batch laoding')
    parser.add_argument('--mem', type=int, default=100, help='memory to use for optuna worker (moab job)')
    parser.add_argument('--skip_train', type=str2bool, default=False, help='wheter to skip training process')
    parser.add_argument('--skip_val', type=str2bool, default=False, help='wheter to skip val stopping')
    parser.add_argument('--tuning_metric', type=str, default='loss', help='metric to use for tuning')

    parser.add_argument('--optuna_metric', type=str, default='loss', help='metric to use for optim optuna')
    parser.add_argument('--optuna_iters', type=int, default=150, help='number of iteration to optimize optuna')
    parser.add_argument('--optuna_schema', type=str, default="unplanned_icu", help='number of iteration to optimize optuna')
    parser.add_argument('--study_id', type=str, default='optuna', help='id for the master who is running the experiments, if interactive optuna else moab job')
    parser.add_argument('--num_workers_per_study', type=int, default=5, help='number of jobs/moab script that will run the experiments')
    parser.add_argument('--exp_id', type=str, help='id for the experiment')

    parser.add_argument('--input_dir', type=str, default='input/', help='dir containing the input file')
    parser.add_argument('--admissions_file', type=str, default='metadata_admissions_train.json', help='file containing the admission dataset')
    parser.add_argument('--admissions_test', type=str, default='metadata_admissions_test.json', help='file containing the test set admission dataset')
    parser.add_argument('--input_file', type=str, default='metadata_diag_biochem.json', help='file containing the admission dataset')
    parser.add_argument('--config_file', type=str, help='file containing the experiment to run dataset')
    parser.add_argument('--date_string', default='None', type=str, help='datestring')
    parser.add_argument('--model', default='deep', type=str, help='which kind od model to run')
    parser.add_argument('--data', default='balanced', type=str, help='data how to balance the data')
    parser.add_argument('--recurrent_layer', default='lstm', type=str, help='which kind od recurrent')
    parser.add_argument('--bidirectional', default=False, type=str2bool, help='flag bidirectional')
    parser.add_argument('--use_gpu', default=False, type=str2bool, help='wheter to use gpu')

    parser.add_argument('--n_splits', type=int, default=3, help='number of splits for cross validation')
    parser.add_argument('--verbose', default=False, type=str2bool, help='training verbose')

    parser.add_argument('--rerun_experiments', type=str2bool, default=False, help='if true the experiments will be overwritten')
    parser.add_argument('--interactive', type=str2bool, default=True, help='run scripts in interactive-mode')
    
    parser.add_argument('--padd_diag', type=int, default=3, help='number of padd diag code')
    parser.add_argument('--padd_biochem', type=int, default=3, help='number of padd diag code')
    parser.add_argument('--padd_notes', type=int, default=3000, help='number of padd diag code')
    parser.add_argument('--min_freq', type=int, default=0, help='minimum frequency of words to be added in the dictionary')
    parser.add_argument('--level_code', default=3, type=int, help='id for the experiment')
    parser.add_argument('-f', '--file', type=str, default='filepath', help='flag for jupyter')
    
    parser.add_argument('--time_to_event', type=str2bool, default=False, help='use deephit approach if true, else multiclass')
    parser.add_argument('--time_windows', type=str, default="30-20-10-5-3-2-1", help='use deephit approach if true, else multiclass')
    parser.add_argument('--baseline_hours', type=int, default=None, help='how many hours after the admission to add data - this is fixed')
    parser.add_argument('--lookahaed', type=int, default=24, help='how many hours after the baseline to check the prediction')
    parser.add_argument('--trigger_time', type=int, default=12, help='how often the model generated a prediction')
    parser.add_argument('--concatenate_time', type=int, default=24, help='range time between two admission to consider them unique')
    parser.add_argument('--minimum_age_at_adm', type=int, default=16, help='minimum age at admission to include the admission in the model')
    parser.add_argument('--force_icu_concatenation', type=str2bool, default=True, help='if true, the icu has to be concat to the previous admission')
    parser.add_argument('--max_days', type=int, default=14, help='when using deephit, the maximum time horizon from the admission')
    parser.add_argument('--combine_metrics', type=str2bool, default=False, help='if multiclass prediction, this flag combines the outcome of death and admission to icu')
    parser.add_argument('--binary_prediction', type=str2bool, default=True, help='if true, death and ICU transfer are considered as a only class')
    parser.add_argument('--diag', type=str2bool, default=True, help='use diags as input')
    parser.add_argument('--biochem', type=str2bool, default=True, help='use biochem as input')
    parser.add_argument('--notes', type=str2bool, default=False, help='use notes as input')
    parser.add_argument('--idx_adm', type=str2bool, default=False, help='use the index of the admissions')
    parser.add_argument('--age_at_adm', type=str2bool, default=False, help='use age as input')
    parser.add_argument('--sex', type=str2bool, default=False, help='use sex as input')
    parser.add_argument('--time', type=str2bool, default=False, help='use timestep as input')
    parser.add_argument('--ews', type=str2bool, default=False, help='use ews as input')
    parser.add_argument('--sql_dataloader', type=str2bool, default=True, help='use sql queries in the get item func')
    parser.add_argument('--filter_ews', type=str2bool, default=False, help='filter only recnum with an EWS record')
    parser.add_argument('--include_percentile', type=str2bool, default=True, help='include percentile value for npu codes')
    parser.add_argument('--top_biochem', type=int, default=250, help='include the 100 most common biochem')
    parser.add_argument('--biochem_bins', type=int, default=10, help='how many bins to split the biochem values')
    
    parser.add_argument('--use_shard', type=str2bool, default=False, help='if true, the data used is sharded in different subfiles')

    return parser

def parse_args(parse_str=None, parent_parser=[]):
    #args for notes
    parser = get_argument_parser(parent_parser)
    args = parser.parse_args(parse_str)
    args.scripts_dir = dirname(dirname(dirname(__file__)))
    args.base_dir = dirname(args.scripts_dir)
    args.input_dir = os.path.join(args.base_dir, "input")
    if not args.exp_id:
        param_string = '.'.join(['{}-{}'.format(k, args.__dict__[k]) for k in sorted(list(args.__dict__.keys()))])
        args.exp_id = hashlib.md5(param_string.encode()).hexdigest()
        
    args = apply_restrictions(args)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['base_dir', 'script_dir', 'input_dir']:
            print("\t{}={}".format(attr.upper(), value))
            
    return args

def parse_nvidia_smi_processes():

    out_dict = defaultdict(list)
    try:
        sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out_str = sp.communicate()
        out_list = out_str[0].decode().split('\n')

        for item in out_list:
            try:
                key, val = item.split(':')
                key, val = key.strip(), val.strip()
                out_dict[key] = val
            except:
                pass
    except:
        #this nodes has no access to nvidia smi command
        pass

    if out_dict['Process ID']:
        print ("""This process {} is already running on the GPU\n Run this command to check who is using the gpu:\tps aux | grep {}""".format(out_dict['Process ID'], out_dict['Process ID']))
    
    return out_dict['Process ID']

def apply_restrictions(args):
    
    if args.biochem_bins not in [10,50,100]:
        raise Exception ("Biochem bins has to be 10,50 or 100 but received {}".format(args.biochem_bins))

    return args
