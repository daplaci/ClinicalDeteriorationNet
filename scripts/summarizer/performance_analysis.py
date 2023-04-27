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
from sklearn.metrics import matthews_corrcoef, \
    precision_recall_curve, roc_curve, \
    auc, roc_auc_score
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
from unplanned_net.utilities.collections import get_best_exp_optuna, get_best_from_auc_table


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def get_metrics_by_time(res, n_boot=1000, match_prob=0, time_resolution=24):
    labels = res["label"].squeeze()
    preds = (res["pred"]).squeeze()
    assert res["pred"].any()
    baselines_res = res["baselines"].squeeze()
    @globalize
    def boot_run(_):
        if match_prob > 0:
            sample = get_match_idxs(labels_, match_prob=match_prob)
        else:
            sample = np.random.choice(preds_.size, preds_.size, replace=True)
        auc_ = roc_auc_score(labels_[sample], preds_[sample])
        p_, r_, t_ = precision_recall_curve(labels_[sample], preds_[sample], pos_label=1)
        auprc_ = auc(r_,p_)
        return auc_, auprc_, h

    aucs, auprcs, baselines = [], [], []
    for h in tqdm(range(0, 24*14, time_resolution)):
        filter_hours = baselines_res==h
        assert filter_hours.any(), f"{h}"
        labels_ = labels[filter_hours]
        preds_ = preds[filter_hours]
        metrics = Pool(32).map(boot_run, range(n_boot))
        aucs_, auprcs_, baselines_= zip(*metrics)
        aucs.extend(aucs_)
        auprcs.extend(auprcs_)
        baselines.extend(baselines_)

    return np.array(aucs), np.array(auprcs),np.array(baselines)


def get_precision_recall_from_res(res, n_boot=1000):
    pred = res["pred"].squeeze()
    label = res["label"].squeeze()

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


def get_auroc_from_res(res, n_boot=1000):
    pred = res["pred"].squeeze()
    label = res["label"].squeeze()

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

def get_match_idxs(label, match_prob=1e-2):
    positive = np.where(label==1)[0].tolist()
    all_negative = np.where(label==0)[0].tolist()
    negative = np.random.choice(all_negative, 
                                size = min(len(all_negative),round((1/match_prob - 1)*len(positive))), 
                                replace = False)
    indexes = np.concatenate((positive, negative), axis=-1)
    return indexes

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='parse args for performance analysis')

    parser.add_argument('--config_file', type=str, 
        required=True, help='skip roc curve generation')

    parser.add_argument('--n_boot', type=int, default=200, 
        help='number of boot strap to run')

    parser.add_argument('--skip_roc_prc', action='store_true', 
        default=False, help='skip roc curve generation')

    parser.add_argument('--time_resolution', type=int, default=12,
        help='time resolution in the metrics plot')

    parser.add_argument('--match_prob', type=float, 
        default=0, help='skip roc curve generation')
    
    parser.add_argument('--skip_visualization', action='store_true', 
        default=False, help='when true just calulcate auroc and auprc CI')
    
    args=parser.parse_args()
        
    base_dir = "/users/projects/clinical_deterioration/output/"
    configs = json.load(open(args.config_file))

    best_exp = []
    for study, fn in zip(configs["study_dirs"], configs["get_best_fn"]):
        output_path = os.path.join(base_dir, study)
        auctable = pd.read_csv(os.path.join(output_path, 'AUC_history_gridsearch.tsv'), sep='\t')
        studies = set(auctable.study_id.tolist())
        try:
            best_trials = eval(fn)(studies, auc_table=auctable)
            best_exp.append(best_trials[0])
        except:
            best_exp.append(fn)

    results = []
    for s,e in zip(configs["study_dirs"], best_exp):
        try:
            results_filename = os.path.join(base_dir, s, "best_weights/{}_1.calibrated.test.pkl".format(e))
            results.append(pickle.load(open(results_filename, 'rb')))
            print (f"Experiment {e} from {s} is loaded")
        except:
            print (f"Warning {results_filename} was not collected. \nTemporarily loading the preformance on internal validation.\n")
            results_filename = os.path.join(base_dir, s, "best_weights/{}_1.pkl".format(e))
            results.append(pickle.load(open(results_filename, 'rb')))

    ## CONSTRUCT DATAFRAME FOR VISUALISATION
    if not args.skip_roc_prc:
        precisions, recalls, true_pos_rate , false_pos_rate, model_roc, model_prc = [], [], [], [], [], []
        for res, study_dir, study, exp_id in zip(results, configs["study_dirs"], configs["study_name"], best_exp):
            print (exp_id)
            ci_auprc, pre, rec , pr_thresholds = get_precision_recall_from_res(res, n_boot=args.n_boot)
            precisions.append(pre)
            recalls.append(rec)
            model_name_prc = study.split(".")[-1].capitalize() + \
                " AUPRC:{:.3f} [{:.3f}-{:.3f}]".format(*ci_auprc)
            print (model_name_prc)
            model_prc.extend([model_name_prc for _ in range(len(pre))])
            
            ci_auroc, fpr, tpr , roc_thresholds = get_auroc_from_res(res, n_boot=args.n_boot)
            true_pos_rate.append(tpr)
            false_pos_rate.append(fpr)
            model_name_roc = study.split(".")[-1].capitalize() + \
                " AUROC:{:.3f} [{:.3f}-{:.3f}]".format(*ci_auroc)
            model_roc.extend([model_name_roc for _ in range(len(tpr))])
            print (model_name_roc)
            saving_metrics = {
                'precision':pre,
                'recall':rec,
                'pr_thresholds':pr_thresholds,
                'tpr':tpr,
                'fpr':fpr,
                'roc_thresholds':roc_thresholds
            }
            metrics_path = os.path.join(base_dir, study_dir, 'best_weights/{}.metrics')
            with open(metrics_path.format(exp_id), 'wb') as f:
                pickle.dump(saving_metrics, f)
        
        precisions = np.concatenate(precisions)
        recalls = np.concatenate(recalls)
        prc_df = pd.DataFrame({"precisions":precisions, 
                            "recalls": recalls,
                            "model":np.asarray(model_prc)}).round(3)

        true_pos_rate = np.concatenate(true_pos_rate)
        false_pos_rate = np.concatenate(false_pos_rate)
        roc_df = pd.DataFrame({"tpr":true_pos_rate, 
                            "fpr":false_pos_rate, 
                            "model":np.asarray(model_roc)}).round(3)

        if not args.skip_visualization:

            ### COMPARE DIFFERENT INPUT AUROC - AUPRC side by side
            fig, axes = plt.subplots(1,2, figsize=(12,6)) 
            sns.lineplot(x='recalls', y='precisions',data=prc_df, ax=axes[0], hue='model')
            sns.lineplot(x='fpr', y='tpr', data=roc_df, ax=axes[1], hue='model')
            axes[0].set(xlabel='Recall', ylabel='Precision', title="Precision-Recall Curve")
            axes[1].set(xlabel='FPR', ylabel='TPR', title="ROC Curve")
            for ax in axes:
                plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
            print ("Saving figure.. ")
            plt.savefig("figures/roc_prc.pdf", bbox_inches='tight', format='pdf')

    baselineDF = pd.DataFrame({"baselines":[], 
                                "auroc":[],
                                "auprc":[]})
                                
    panels_prob = [0, 1e-2, 0.1, 0.5]
    fig, axes = plt.subplots(1,len(panels_prob), figsize = (18,12))
    
    ageDF = pd.DataFrame({"baselines":[], 
                                "auroc":[],
                                "auprc":[],
                                'age':[]})
                                
    panels_age = np.linspace(16, 100, 5)
    panels_age = np.rint(panels_age).astype('int32')
    
    for age_idx in range(len(panels_age)-1):
        min_age = panels_age[age_idx]
        max_age = panels_age[age_idx+1]
        age_results = {k:v[np.logical_and(results[-1]['age_at_adm']>min_age, results[-1]['age_at_adm']<=max_age)] for k,v in results[-1].items()}
        aucs , auprc, baselines = get_metrics_by_time(age_results, #change results here if you do not want the metrics on the last folder
                                                                    n_boot=args.n_boot, 
                                                                    time_resolution=args.time_resolution)
    
        df = pd.DataFrame({"baselines":baselines, 
                                "auroc":aucs,
                                "auprc":auprc})
        age_str = '-'.join([str(min_age), str(max_age)])
        df['age'] = age_str
        ageDF = ageDF.append(df, ignore_index=True)

        auprc_ci, *_ = get_precision_recall_from_res(age_results, 
                                    n_boot=args.n_boot)
        auroc_ci, *_ = get_auroc_from_res(age_results, 
                                            n_boot=args.n_boot)
        
        saving_metrics.update(
        {"age_{}_prc".format(age_str):auprc_ci,
        "age_{}_roc".format(age_str):auroc_ci,}
        )
    
    if not args.skip_visualization:
        fig, axes = plt.subplots(1,2, figsize = (12,6))
        sns.lineplot(data=ageDF, x='baselines', y='auprc', hue='age', ax=axes[0])
        sns.lineplot(data=ageDF, x='baselines', y='auroc', hue='age', ax=axes[1])
        axes[0].set(xlabel='Hours after admission', ylabel='AUPRC', title="AUPRC - Age stratification")
        axes[1].set(xlabel='Hours after admission', ylabel='AUROC', title="AUROC - Age stratification")
        plt.savefig("figures/auc_prc_age.pdf", bbox_inches='tight', format='pdf')
        print ("Saving age figure.. ")
        plt.close()
    
    sexDF = pd.DataFrame({"baselines":[], 
                                "auroc":[],
                                "auprc":[],
                                'sex':[]})
                                
    panels_sex = np.unique(results[-1]['sex'])
    
    for sex in panels_sex:
        sex_results = {k:v[results[-1]['sex']==sex] for k,v in results[-1].items()}
        aucs , auprc, baselines = get_metrics_by_time(sex_results, #change results here if you do not want the metrics on the last folder
                                                    n_boot=args.n_boot, 
                                                    time_resolution=args.time_resolution)
        df = pd.DataFrame({"baselines":baselines, 
                                "auroc":aucs,
                                "auprc":auprc})
        if sex>0.5:
            sex_str = "Male"
        else:
            sex_str = "Female"
        df['sex'] = sex_str
        sexDF = sexDF.append(df, ignore_index=True)

        auprc_ci, *_ = get_precision_recall_from_res(sex_results, 
                                    n_boot=args.n_boot)
        auroc_ci, *_ = get_auroc_from_res(sex_results, 
                                            n_boot=args.n_boot)
        
        saving_metrics.update(
        {"sex_{}_prc".format(sex_str):auprc_ci,
        "sex_{}_roc".format(sex_str):auroc_ci,}
        )
    
    if not args.skip_visualization:
        fig, axes = plt.subplots(1,2, figsize = (12,6))
        sns.lineplot(data=sexDF, x='baselines', y='auprc', hue='sex', ax=axes[0])
        sns.lineplot(data=sexDF, x='baselines', y='auroc', hue='sex', ax=axes[1])
        axes[0].set(xlabel='Hours after admission', ylabel='AUPRC', title="AUPRC - Sex stratification")
        axes[1].set(xlabel='Hours after admission', ylabel='AUROC', title="AUROC - Sex stratification")
        plt.savefig("figures/auc_prc_sex.pdf", bbox_inches='tight', format='pdf')
        print ("Saving figure sex.. ")

    with open(metrics_path.format(exp_id), 'wb') as f:
        pickle.dump(saving_metrics, f)