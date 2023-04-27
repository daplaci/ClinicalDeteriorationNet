import numpy as np
from unplanned_net.dataset.datasource import NumericalDataSource
import unplanned_net.dataset.admissions_dataset as admissions_dataset

class EhrAdmissionsDataset(admissions_dataset.GeneralAdmissionsDataset):
    def __init__(self, adm_data, ehr_data, t_person, split_group, args, datasources=None, stats=None):
        super(EhrAdmissionsDataset, self).__init__(adm_data, ehr_data, t_person, split_group, args, datasources, stats)
        
        self.rel_time_bins = list(map(lambda x: int(x)*(-3600*24*365), 
                                      args.time_windows.split('-')))

        self.rel_time_bins += [3600*n for n in range(0, 30*24+1, self.args.trigger_time)]
        self.timepoints = len(args.time_windows.split('-'))

    def get_bins(self, t_adm):
        bins = list([(t_adm+n_hours) for n_hours in self.rel_time_bins])
        return bins

    def __get_data_for_stats__(self, pid, idx_adm, baseline, datasource = 'diag'):

        t_adm = self.adm_data[pid]['admissions'][idx_adm]["adm_datetime"]
        time_limit = t_adm + baseline*3600

        if self.args.sql_dataloader:
            pass
        else:
            raise NotImplementedError
        
        for ds in self.datasources:
            if ds.name == datasource:
                data = ds.get_data_for_stats(self.con, pid, time_limit, t_adm)
                return data
        
    def __getitem__(self, index):
        inputs = {}
        pid, idx_adm, label  = self.valid_indexes[index]["pid"], self.valid_indexes[index]["idx_adm"], self.valid_indexes[index]["label"]
        baseline = self.valid_indexes[index]["baseline"]
        time_to_event = self.valid_indexes[index]["time_to_event"]

        t_adm = self.adm_data[pid]['admissions'][idx_adm]["adm_datetime"]
        time_limit = t_adm + baseline*3600

        if self.args.sql_dataloader:
            pass
        else:
            raise NotImplementedError
        
        inputs_text = {}
        inputs = {}
        seq_lengths = {}

        for ds in self.datasources:
            if isinstance(ds, NumericalDataSource):
                inputs_text[ds.name] = []
                inputs[ds.name]  = ds.get2d_input(self.valid_indexes[index][ds.name])
                seq_lengths[ds.name] = [] 
            else:
                inputs_text[ds.name], inputs[ds.name], seq_lengths[ds.name] = ds.get2d_input(self.con, pid, time_limit)

        baseline = self.timepoints + baseline/self.args.trigger_time
        return index, inputs_text, inputs, seq_lengths, baseline,  np.float32(label), np.float32(time_to_event)

