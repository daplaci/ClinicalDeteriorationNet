import sys
import os
import json
import pandas as pd
import numpy as np
import pickle
import argparse
import math
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score \
    precision_recall_curve, roc_curve, auc
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from unplanned_net.utilities.collections import get_best_exp_optuna, get_successfull_trials


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
 
def get_metrics_from_res(res, interval=0, n_boot=1000, trigger_time=12):
    labels = res["label"].squeeze()
    preds = vsigmoid(res["pred"]).squeeze()
    baselines_res = res["baselines"].squeeze()

    aucs, mccs, auprcs, baselines, incidences = [], [], [], [], []
    for h in range(0, 24*30, trigger_time):
        filter_hours = np.logical_and(baselines_res>=h, baselines_res<=(h+interval))
        labels_ = labels[filter_hours]
        preds_ = preds[filter_hours]
        for _ in range(n_boot):
            sample = np.random.choice(preds_.size, preds_.size, replace=True)
            auc_ = roc_auc_score(labels_[sample], preds_[sample])
            p_, r_, t_ = precision_recall_curve(labels_[sample], preds_[sample], pos_label=1)
            score = 2*(p_*r_)/(p_+r_) #f1_score 
            threshold = t_[np.argmax(score)]
            mcc_ = matthews_corrcoef(labels_[sample], (preds_[sample]>threshold).astype("int32"))
            aucs.append(auc_)
            auprcs.append(auc(r_,p_))
            mccs.append(mcc_)
            baselines.append(h)
            incidence = sum(labels_[sample])/len(labels_[sample])
            incidences.append(incidence)

    return np.array(aucs), np.array(mccs), np.array(auprcs),np.array(baselines), np.array(incidences)


def get_precision_recall_from_res(res, h, interval=0, n_boot=1000):
    baselines = res["baselines"]
    filter_hours = np.logical_and(baselines>=h, baselines<=(h+interval))
    pred = vsigmoid(res["pred"][filter_hours]).squeeze()
    label = res["label"][filter_hours].squeeze()

    precisions, recalls, thresholds = precision_recall_curve(
            label, pred, pos_label=1)
    auprcs = []
    for _ in range(n_boot):
        sample = np.random.choice(pred.size, pred.size, replace=True) 
        precisions_, recalls_, _ = precision_recall_curve(
            label[sample], pred[sample], pos_label=1
        )
        auprcs.append(auc(recalls_, precisions_))
    auprc_ci = mean_confidence_interval(auprcs)
    return auprc_ci, precisions, recalls, thresholds


def get_auroc_from_res(res, h, interval=0, n_boot=1000):
    baselines = res["baselines"]
    filter_hours = np.logical_and(baselines>=h, baselines<=(h+interval))
    pred = vsigmoid(res["pred"][filter_hours]).squeeze()
    label = res["label"][filter_hours].squeeze()

    fpr, tpr, thresholds = roc_curve(
            label, pred, pos_label=1
        ) 
    aurocs = []
    for _ in range(n_boot):
        sample = np.random.choice(pred.size, pred.size, replace=True) 
        fpr_, tpr_, _ = roc_curve(
            label[sample], pred[sample], pos_label=1
        )
        aurocs.append(auc(fpr_, tpr_))
    auroc_ci = mean_confidence_interval(aurocs)
    return auroc_ci, fpr, tpr, thresholds

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    hm, m, mh = np.percentile(a, (2.5, 50, 97.5))
    return m, hm, mh

if __name__=="__main__":
    
    vsigmoid = np.vectorize(sigmoid)
    results_dir = "results/"
    study_id = "99130319985866034e8c675efffaf041"

    trials = get_successfull_trials(study_id)

    baselineDF = pd.DataFrame({"baselines":[], 
                            "auroc":[],
                            "mcc":[],
                            "auprc":[],
                            "lookahaed":[],
                            "trigger_time":[]})
    for trial in tqdm(trials):
        #define parameter
        exp_id = trial.user_attrs['exp_id']
        #load results
        try:
            results = pickle.load(open(os.path.join(results_dir, f"{exp_id}_1.calibrated.test.pkl"), 'rb'))
        except FileNotFoundError as e:
            print (f"missing experiment {exp_id}")
            continue
        #put results in visualization df
        aucs , mccs, auprc, baselines, incidence = get_metrics_from_res(results, #change results here if you do not want the metrics on the last folder
                                                                    interval=0,
                                                                    n_boot=200,  ##TODO 
                                                                    trigger_time=trial.params['trigger_time'])
        
        lookahaed_ = np.array([trial.params['lookahaed'] for _ in range(len(incidence))])
        trigger_time_ = np.array([trial.params['trigger_time'] for _ in range(len(incidence))])
    
        df = pd.DataFrame({"baselines":baselines, 
                                "auroc":aucs,
                                "mcc":mccs,
                                "auprc":auprc,
                                "lookahaed":lookahaed_, 
                                "trigger_time":trigger_time_})

        baselineDF = baselineDF.append(df, ignore_index=True)

    fig, axes = plt.subplots(1,2, figsize = (12,6))
    sns.lineplot(data=baselineDF, x="baselines", y="auroc", ax=axes[0])
    axes[0].set_title("AUROC", fontweight='bold')
    axes[0].set(xlabel="Hours after admission", ylabel="AUROC")
    sns.lineplot(data=baselineDF, x="baselines", y="auprc", ax=axes[1])
    axes[1].set_title("AUPRC", fontweight='bold')
    axes[1].set(xlabel="Hours after admission", ylabel="AUPRC")

    plt.savefig("figures/lookahaed_trigger.pdf", bbox_inches='tight', format='pdf')
    baselineDF.to_csv("figures/metrics_at_lookahaed_trigger.tsv", index=False, sep='\t')