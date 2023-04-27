from collections import defaultdict
import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import uuid
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import math
from multiprocessing import Pool
PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
sys.path.append(PATH_TO_LIB)
from unplanned_net.utilities.collections import get_params_from_successfull_trial
from unplanned_net.utilities.collections import get_successfull_trials

def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def get_metrics_from_res(res, n_boot=1000):
    pred = res["pred"].squeeze()
    label = res["label"].squeeze()

    __precisions__ = defaultdict(list)
    __recalls__ = defaultdict(list)
    __specificities__ = defaultdict(list)
    
    for _ in tqdm(range(n_boot)):
        sample = np.random.choice(pred.size, pred.size, replace=True) 
        precisions_, recalls_, thresholds_ = precision_recall_curve(
            label[sample], pred[sample], pos_label=1
        )

        for risk in RISKS:
            __precisions__[risk].append(precisions_[:-1][thresholds_<risk][-1])
            __recalls__[risk].append(recalls_[:-1][thresholds_<risk][-1])
        
        specificities_, _, thresholds_ = roc_curve(
            label[sample], pred[sample], pos_label=1
        )      
        for risk in RISKS:
            __specificities__[risk].append(1 - specificities_[thresholds_<risk][0])  
    
    precisions = dict.fromkeys(__precisions__)
    recalls = dict.fromkeys(__recalls__)
    specificities = dict.fromkeys(__specificities__)
 
    for risk in RISKS: 
        precisions[risk] = mean_confidence_interval(__precisions__[risk])
        recalls[risk] = mean_confidence_interval(__recalls__[risk])
        specificities[risk] = mean_confidence_interval(__specificities__[risk])
    
    return precisions, recalls, specificities

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    hm, m, mh = np.percentile(a, (2.5, 50, 97.5))
    return m, hm, mh


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='parse args for performance analysis')

    parser.add_argument('--output_path', type=str, 
        required=True, help='path to use to extract the metrics')
    parser.add_argument('--study_id', type=str, 
        required=True, help='study to get experiments')
    parser.add_argument('--n_boot', type=int, default=200, 
        help='number of boot strap to run')

    args=parser.parse_args()
    
    base_dir = "/users/projects/clinical_deterioration/output/"
    RISKS = [1/100, 5/100, 10/100, 20/100, 50/100]
    
    trials = get_successfull_trials(args.study_id)
    exp_ids = set([trial.user_attrs['exp_id'] for trial in trials])
    ## CONSTRUCT DATAFRAME FOR VISUALISATION
    metric_records = []
    for exp_id in exp_ids:
        result_filename = os.path.join(
            args.output_path,
            "best_weights/{}_1.calibrated.test.pkl".format(exp_id))
        result = pickle.load(open(result_filename, "rb")) 
        print (exp_id)
        params = get_params_from_successfull_trial(args.study_id, exp_id)
        precisions, recalls, specificities = get_metrics_from_res(result, n_boot=args.n_boot)
        for risk in RISKS:
            metric_records.append(
                (exp_id,
                args.study_id,
                params["trigger_time"],
                params["lookahaed"],
                risk,
                precisions[risk],
                recalls[risk],
                specificities[risk])
            )
    df = pd.DataFrame.from_records(
        metric_records, 
        columns=["exp_id", 
                "study", 
                "Frequency of assessment", 
                "Prediction window", 
                "Risk", 
                "Precision/PPV", 
                "Recall/TPR", 
                "Specificity"])

    df.to_csv('figures/pdig_revision/supplementary_metrics.tsv', sep='\t', index=False)

