import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from unplanned_net.utilities.vocab import *
from unplanned_net.utilities.database import MyDB
import unplanned_net.dataset.collate as collate
from unplanned_net.dataset.ehr_admissions_dataset import EhrAdmissionsDataset

class CustomDataLoader():
    def __init__(self, *args,  force_positive_occurrence=False, **kwargs):
        self.force_positive_occurrence = force_positive_occurrence
        self.dataloader = data.DataLoader(*args, **kwargs)
         
    def __len__(self):
        return self.dataloader.__len__()

    def __iter__(self):
        self.iter = iter(self.dataloader)
        return self

    def __next__(self):
        while True:
            batch = next(self.iter)
            _, _, _, _, _, label, _ = batch
            if (not self.force_positive_occurrence or sum(label)>0):
                break
        return batch

class CustomWeightedRandomSampler(data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.con = MyDB()

def get_dataclasses(adm_data, data_diag, t_person, split_group, args, 
                    datasources=None, full_validation=False, 
                    use_weights_sampler=False, stats=None, 
                    force_positive_occurrence=False):

    print ("Building {} dataset and dataloader..\n".format(split_group))
    
    dataset = EhrAdmissionsDataset(adm_data, data_diag, t_person, split_group, args, 
                                    datasources=datasources, stats=stats)
    collator = collate.default_collate

    if use_weights_sampler:
        if len(dataset.weights) < 2**24:
            sampler = data.sampler.WeightedRandomSampler(dataset.weights, 
                                                    len(dataset), 
                                                    replacement= True,)
        else:
            print ("Using Custom weights sampler")
            sampler = CustomWeightedRandomSampler(dataset.weights, 
                                                    len(dataset), 
                                                    replacement= True,)     
    else: 
        sampler = None    
        
    if split_group == 'train':
        batch_size = args.train_batch if not full_validation else len(dataset)
    else:
        batch_size = args.val_batch if not full_validation else len(dataset)

    if args.num_workers > 0:
        data_generator = CustomDataLoader(dataset, 
                                        batch_size=batch_size, num_workers=args.num_workers,
                                        sampler=sampler, 
                                        collate_fn=collator, worker_init_fn=worker_init_fn, 
                                        persistent_workers=True, prefetch_factor=4, 
                                        force_positive_occurrence=force_positive_occurrence)
    else:
        dataset.con = MyDB()
        data_generator = CustomDataLoader(dataset, 
                                        batch_size=batch_size, num_workers=args.num_workers,
                                        sampler=sampler, collate_fn=collator, 
                                        force_positive_occurrence=force_positive_occurrence)
    return dataset, data_generator

def get_input_data(args):
    adm_data = json.load(open(os.path.join(args.input_dir, args.admissions_file),'r'))
    print ("Laoded admission data")

    t_person =  pd.read_csv("t_person.tsv", sep='\t').astype(str)
    print ("Loaded t_person")

    if not args.use_shard and not args.sql_dataloader:
        scratch_path = '/scratch/'
        if os.path.exists(scratch_path) and args.input_file in os.listdir(scratch_path):
            input_dir = scratch_path
        else:
            input_dir = args.input_dir
        print ("Load EHR data from {}".format(input_dir))
        ehr_data = json.load(open(os.path.join(input_dir, args.input_file),'r'))
    
        assert not args.notes , ("Notes is true but shard is false. If you want to use the Notes \
                                without shard, you have to generate the file")
    else:
        ehr_data = None
    print ("Data Generated")
    return adm_data, ehr_data, t_person

