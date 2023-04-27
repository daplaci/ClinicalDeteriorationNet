from utilities.date import *
from torch.utils import data
import json
import pandas as pd
import pdb
import tqdm
import utilities.gen_mask as gen_mask
import numpy as np
import pickle
from utilities.vocab import *
import dataset.admissions_dataset as admissions_dataset
from dataset.admissions_dataset import sort_time
from utilities.time import timeit

class MlpAdmissionsDataset(admissions_dataset.GeneralAdmissionsDataset):
    def __init__(self, adm_data, ehr_data, t_person, split_group, args, datasources=None, stats=None):
        super(MlpAdmissionsDataset, self).__init__(adm_data, ehr_data, t_person, split_group, args, stats)

    def __getitem__(self, index):
        inputs = {}
        pid, idx_adm, label = self.valid_indexes[index]["pid"], self.valid_indexes[index]["idx_adm"], self.valid_indexes[index]["label"] 

        t_adm = self.adm_data[pid]['admissions'][idx_adm]["adm_datetime"]
        time_limit = t_adm + self.args.baseline_hours*3600

        label = np.float32(label!=2) if self.args.binary_prediction else label

        if self.args.sql_dataloader:
            item_ehr_data = self.con.create_item_ehr_from_sql_queries(pid, time_limit)
            time = sort_time(item_ehr_data[pid], time_limit=time_limit) 
        else:
            assert self.ehr_data
            item_ehr_data = self.ehr_data
            time = sort_time(item_ehr_data[pid], time_limit=time_limit) 
            time = map(str, time)

        inputs_text = {}
        inputs = {}
        
        for ds in self.datasources:
            inputs_text[ds.name], inputs[ds.name] = ds.get1d_input(item_ehr_data[pid], self.valid_indexes[index], time)

        time_to_event = self.valid_indexes[index]["time_to_event"]

        return index, inputs_text, inputs, [], [],  np.float32(label), np.float32(time_to_event)