import sys
import os
import numpy as np
import pandas as pd
import traceback
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from copy import deepcopy
import hashlib
import torch
import pickle
from tqdm import tqdm
from unplanned_net.run.learn import generate_metrics
import unplanned_net.utilities.metrics as metrics
import unplanned_net.run.learn as learn

stripper = lambda x : str(x[0]) if ((type(x) is tuple or type(x) is list) and len(x)==1)  else str(x)


class DF_writer():
    def __init__(self, ):
        self.AUCheader = list(pd.read_csv('AUC_history_gridsearch.tsv', sep='\t', nrows=1).columns.values)
        self.CVheader = list(pd.read_csv('CV_history_gridsearch.tsv', sep='\t', nrows=1).columns.values)
        self.progressHeader = list(pd.read_csv('progress.log', sep='\t', nrows=1).columns.values)
        self.errorHeader = list(pd.read_csv('error.log', sep='\t', nrows=1).columns.values)

    def write_progress(self):
        print("Writing the progress of this Job")
        progressDF = pd.DataFrame(columns=self.progressHeader)
        progressDF = progressDF.append({'completed' : 'final'}, ignore_index = True)
        progressDF.to_csv('progress.log', header=None, index=False, 
                                    sep='\t', mode='a', columns = progressDF.columns)
        
    def write_error(self):
        print("\n*****Error.. Writing the error for this Job*****\n")
        var = traceback.format_exc()
        errorDF = pd.DataFrame(columns=self.errorHeader)
        errorDF = errorDF.append({'error': var,'args' : self.AUCheader}, ignore_index = True)
        errorDF.to_csv('error.log', header=None, index=False, sep='\t', mode='a', columns = errorDF.columns)
    
    def write_auc(self, args):
        args_to_df = {k:stripper(v) for k,v in args.__dict__.items() if k in self.AUCheader}
        print("\nSaving this params to AUC table:")
        for attr, value in sorted(args_to_df.items()):
            print("\t{}={}".format(attr.upper(), value))
        #assert (len(args_to_df)==len(self.AUCheader))
        AUChistory = pd.DataFrame(columns=self.AUCheader)
        AUChistory = AUChistory.append(args_to_df, ignore_index = True)
        AUChistory.to_csv('AUC_history_gridsearch.tsv', index=False,sep='\t', mode='a', columns = AUChistory.columns, header=None)

    def write_history(self, args, train_hist):
        history = {k: [] for k in self.CVheader}
        
        for dataset, iterstat in train_hist.epoch_stats.items():
            for metric, metric_value in {**{'loss':iterstat.loss}, **iterstat.metrics}.items():
                if dataset == 'train':
                    key = metric
                else:
                    key = "{}_{}".format(dataset, metric)
                if metric not in history:
                    print ("******\nWarnign: this {} is missing in CVhistory".format(metric))
                    continue
                num_epochs = len(metric_value)
                history[key] = metric_value
        
        for k in history:
            if not history[k] and k in self.CVheader:
                history[k] = [i if k not in args.__dict__ else stripper(args.__dict__[k])  
                            for i in range(num_epochs)]

        model_history = pd.DataFrame.from_dict(history)
        model_history.to_csv('CV_history_gridsearch.tsv', index=False, sep='\t', mode='a', columns = model_history.columns, header=None)
    
    def is_duplicate(self, args):
        args_to_df = {k:stripper(v) for k,v in args.__dict__.items() if k in self.AUCheader}
        AUChistory = pd.DataFrame(columns=list(args_to_df.keys()))
        AUChistory = AUChistory.append(args_to_df, ignore_index = True)
        AUCdataframe = pd.read_csv('AUC_history_gridsearch.tsv', usecols=list(args_to_df.keys()), sep='\t')
        AUCdataframe = AUCdataframe.drop_duplicates(keep='first')
        AUCtest_df = pd.concat([AUChistory, AUCdataframe], join='inner')
        if AUCtest_df.duplicated().any():
            print ("This args config already exists:\n", AUCtest_df.loc[AUCtest_df.duplicated(keep='first')].values)
        return AUCtest_df.duplicated().any()

    def save_pred(self, pred, label, baselines, age_at_adm, args, cm=None):
        result = {'pred':pred, 'label':label, 'baselines':baselines, 'age_at_adm':age_at_adm}
        result.update(args.__dict__)
        id_ = args.weights_path.replace(".pt", "")
        with open("{}.pkl".format(id_), 'wb') as f:
            pickle.dump(result, f)
        if cm is not None:
            np.save("{}_cm.npy".format(id_), cm)

    def update_train_results(self, train_loader, val_loader, model, optimizer, train_hist, args):                            
        
        print ("Running eval and saving the results..\n\n")

        model.load_state_dict(torch.load(args.weights_path)['model_state_dict']) 
        optimizer.load_state_dict(torch.load(args.weights_path)['optimizer_state_dict'])

        args_to_writer =deepcopy(args)
        if train_hist:
            self.write_history(args_to_writer, train_hist)

        train_gen = train_loader.__iter__()
        val_gen = val_loader.__iter__()


        train_metrics = []
        for _ in range(args.steps_per_epoch_train):
            batch = next(train_gen)
            learn.val_batch (batch, model, args, stats_keeper=train_metrics)
        
        for m in train_metrics:
            for metric_name in m :
                args_to_writer.__dict__[metric_name] = np.nanmean(np.asarray(m[metric_name]))
        
        pred_list,label_list,baselines_list, age_at_adm_list = [], [], [],[]

        for batch in tqdm(val_gen):
            batch_results  = learn.val_batch(batch, model, args,)
            pred_list.append(batch_results["output"].cpu().data.numpy()) 
            label_list.append(batch_results["label"].cpu().data.numpy())
            baselines = batch_results["baselines"].cpu().data.numpy() - len(args.time_windows.split('-'))
            baselines_list.append(baselines*args.trigger_time)
            if "age_at_adm" in batch_results["inputs"]:
                age_at_adm_list.append(batch_results["inputs"]["age_at_adm"].cpu().data.numpy())
        
        pred = np.concatenate(pred_list)
        label = np.concatenate(label_list)
        baselines = np.concatenate(baselines_list) if baselines_list else baselines_list
        age_at_adm = np.concatenate(age_at_adm_list) if age_at_adm_list else age_at_adm_list
    
        output = pred  
        metrics = generate_metrics(output, label)
        args_to_writer.__dict__.update({'val_'+k:v for k,v in metrics.items()})

        self.write_auc(args_to_writer)
        try:
            cm = confusion_matrix(label-1, output)
        except:
            cm = None

        self.save_pred(pred, label, baselines, age_at_adm, args, cm=cm)

