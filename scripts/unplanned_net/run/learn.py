import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from unplanned_net.utilities.time import timeit
import unplanned_net.utilities.losses as losses
np.seterr('ignore')

class ModelCheckpoint():
    
    def __init__(self, metric='mcc', model_save_path=None):
        self.metric = metric
        if metric == 'mcc':
            self.optim = 0
        else:
            self.optim = np.inf
        self.best_model_name = None
        self.model_save_path = model_save_path
        self.best_epoch = 0
        self.best_state = None
        print ("Training w tuning metric: {}".format(metric))

    def is_best(self, current_metric):
        
        if self.metric == 'loss':
            if current_metric < self.optim:
                print ("""status: val_{} improved from {} \
                to {}\n""".format(self.metric, self.optim, current_metric))
                self.optim = current_metric
                return True
            else:
                print ("""status: val_{} did not \
                improved from {} since epoch {}\n""".format(self.metric,self.optim,self.best_epoch))
        
        if self.metric in ['mcc', 'c_index']:
            if current_metric > self.optim:
                print ("status: val_{} improved from {} to {}".format(self.metric,self.optim,current_metric))
                self.optim = current_metric
                return True
            else:
                print ("status:{} did not improved from {} since epoch {}\n".format(self.metric,self.optim,self.best_epoch))

    def store_model(self, model, optimizer, iteration):
        self.best_epoch = iteration
        print("Storing model.. ")
        self.best_state = {
            'iter': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_loss': self.optim
            } 

    def save_model(self):
        print (f"Saving Model from iteration .. {self.best_epoch}")
        torch.save(self.best_state, self.model_save_path)

class IteratorStats():
    def __init__ (self, dataset):
        self.dataset = dataset
        self.metrics = defaultdict(list)
        self.loss = []

    def update_loss(self, loss):
        self.loss.append(loss)
    
    def update_metrics(self, metrics):
        if metrics:
            for metric_name, metric_value in metrics.items():
                self.metrics[metric_name].append(metric_value)
    
    def avg_loss(self):
        return np.nanmean(self.loss)
    
    def avg_metrics(self,):
        m = {k:np.nanmean(self.metrics[k]) for k in self.metrics}
        return m
    
    def latest_update(self):
        m = '\t'.join(["{}: {:2.3f}".format(k,v[-1]) for k,v in self.metrics.items()])
        s = "loss {:2.3f}\t{}\n".format(self.loss[-1], m)
        return s

    def __len__(self):
        return len(self.loss)

class StatsKeeper():
    def __init__ (self, n_epochs, train_update_per_epoch=1, val_update_per_epoch=1):
        self.epoch_stats= {'train': IteratorStats('train'), 'val':IteratorStats('val')}

        self.batch_stats = {'train':IteratorStats('train'), 'val': IteratorStats('val')}

        self.n_epochs = n_epochs
        self.update_per_epoch = {'train':train_update_per_epoch,'val':val_update_per_epoch} #
        self.best_epoch = 0

    def update_step(self, loss, metrics, dataset='train'):

        self.batch_stats[dataset].update_loss(loss)
        self.batch_stats[dataset].update_metrics(metrics)

        if len(self.batch_stats[dataset]) == self.update_per_epoch[dataset]:

            epoch_loss = self.batch_stats[dataset].avg_loss()
            self.epoch_stats[dataset].update_loss(epoch_loss)
            epoch_metrics = self.batch_stats[dataset].avg_metrics()
            self.epoch_stats[dataset].update_metrics(epoch_metrics)

            self.batch_stats = {'train':IteratorStats('train'), 'val': IteratorStats('val')}

    def print_epoch_stats(self, epoch):
        for dataset, iteratorstats in self.epoch_stats.items():
            if len(iteratorstats) > 0:
                print("{} epoch {}:\t".format(dataset, epoch))
                print(iteratorstats.latest_update())


def run_epochs(train_loader, val_loader, model, optimizer, args):

    model_checkpoint = ModelCheckpoint(args.tuning_metric, args.weights_path)
    stats_keeper = StatsKeeper(args.n_epochs, train_update_per_epoch=args.steps_per_epoch_train, val_update_per_epoch=args.steps_per_epoch_val)

    train_gen = train_loader.__iter__()
    val_gen = val_loader.__iter__()

    for epoch in range(args.n_epochs):

        for _ in tqdm(range(args.steps_per_epoch_train)):
            try:
                batch = next(train_gen)
            except StopIteration:
                print("Train loader exausted.. reinitializing it")
                train_gen = train_loader.__iter__()
                batch = next(train_gen)
            
            train_batch(batch, 
                model, 
                optimizer, 
                args, 
                stats_keeper)
        
        for _ in range(args.steps_per_epoch_val):
            try:
                batch = next(val_gen)
            except StopIteration:
                print ("Val loader exausted.. reinitializing it")
                val_gen = val_loader.__iter__()
                batch = next(val_gen)
            
            val_batch(batch,
                    model,
                    args,
                    stats_keeper)

        stats_keeper.print_epoch_stats(epoch)
        
        is_best = False
        
        if args.tuning_metric == 'loss':
            is_best = model_checkpoint.is_best(stats_keeper.epoch_stats['val'].loss[-1])
        
        elif (args.tuning_metric in stats_keeper.epoch_stats['val'].metrics):
            is_best = model_checkpoint.is_best(stats_keeper.epoch_stats['val'].metrics[args.tuning_metric][-1])

        if is_best or epoch==0:    
            model_checkpoint.store_model(model, optimizer, epoch)
            stats_keeper.best_epoch = epoch

        if (epoch - model_checkpoint.best_epoch) > args.patience:
            break
    
    model_checkpoint.save_model()
    return stats_keeper

def train_batch(batch, model, optimizer, args, stats_keeper=None):
    model.train()

    batch = batch_to_device(batch, model.device)
    _, _, inputs, seq_lens, baselines,  label, time_to_event = batch
    optimizer.zero_grad()
    output = model(inputs, seq_lens, baselines)
    loss = losses.total_loss(output, seq_lens, baselines, label, time_to_event, model.device, args)

    loss.backward()
    optimizer.step()
    
    if stats_keeper:
        metrics = {}
        #metrics = generate_metrics(output, label)
        stats_keeper.update_step(loss.item(), metrics, 'train')

def val_batch(batch, model, args, stats_keeper=None):
    model.eval()
    
    batch = batch_to_device(batch, model.device)
    _, _, inputs, seq_lens, baselines,  label, time_to_event = batch
    output = model(inputs, seq_lens, baselines)
    loss = losses.total_loss(output, seq_lens, baselines, label, time_to_event, model.device, args)
    
    if type(stats_keeper) is list:
        metrics = generate_metrics(output, label)
        stats_keeper.append(metrics)

    elif isinstance(stats_keeper, StatsKeeper):
        #metrics = generate_metrics(output, label)
        metrics = {}
        stats_keeper.update_step(loss.item(), metrics, 'val')
    else:
        return {"output":output, 
                "label":label, 
                "baselines":baselines, 
                "inputs":inputs, 
                "time_to_event":time_to_event}
    

def batch_to_device(batch, device):
    batch_device = []
    for i, el in enumerate(batch):
        if type(el).__module__=='torch':
            batch_device.append(el.to(device))
        elif type(el) is dict:
            for k,v in el.items():
                if type(v).__module__=='torch':
                    el[k] = v.to(device)
            batch_device.append(el)
        else:
            batch_device.append(el)
    return batch_device

def generate_metrics(output, label):
    
    metrics_dict = defaultdict(lambda : 0)
    
    if isinstance(label, torch.Tensor):
        label = label.cpu().data.numpy()
    if isinstance(output, torch.Tensor):
        output = output.cpu().data.numpy()
    
    try:
        auroc = roc_auc_score(label, output)
        mcc = matthews_corrcoef(label, (output>0).astype("int32"))
        precision = precision_score(label, (output>0).astype("int32"), zero_division=0)
        recall = recall_score(label, (output>0).astype("int32"), zero_division=0)
        p,r,_ = precision_recall_curve(label, output)
        auprc = auc(r, p)
    
    except ValueError:
        auroc = np.nan
        mcc = np.nan
        precision = np.nan
        recall = np.nan
        auprc = np.nan
    
    metrics_dict.update({'auc':auroc}) 
    metrics_dict.update({'mcc':mcc})
    metrics_dict.update({'precision':precision})
    metrics_dict.update({'recall':recall})
    metrics_dict.update({'auprc':auprc})         

    return metrics_dict

