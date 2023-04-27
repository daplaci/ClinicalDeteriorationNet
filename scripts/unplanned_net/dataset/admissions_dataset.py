import json
from typing import List, Union
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils import data
from collections import Counter, defaultdict
from unplanned_net.dataset.datasource import NumericalDataSource, TextDataSource 
from unplanned_net.utilities.date import parse_date

def sort_time(l, time_limit=None):
    if time_limit:
        time = [float(el) for el in l.keys() if float(el)<=time_limit]
    else:
        time = list(map(float, l.keys()))
    time = sorted(time, reverse=False) # it returns float list of time
    return time

class GeneralAdmissionsDataset(data.Dataset):

    def __init__(self, adm_data: dict, 
        ehr_data:Union[dict, None], 
        t_person: pd.DataFrame, 
        split_group:str, 
        args, 
        datasources: List[Union[TextDataSource, NumericalDataSource]], 
        stats=None):
        super(GeneralAdmissionsDataset, self).__init__()

        self.args  = args
        self.adm_data = adm_data
        self.ehr_data = ehr_data
        self.split_group = split_group
            
        self.filter_summary = {
            "Tperson filter": 0,
            "Tperson filter ICU": 0, 
            "Age filter" : 0, 
            "Age filter ICU": 0, 
            "Missing Concatenation filter ICU": 0, 
            "Raw ICU admissions":0,
            "Concatenated admissions":0, 
            "Timelimit admissions":0, 
            "Timelimit admissions ICU transfer":0,
            "Death within 14d from discharge":0
            }

        self.type_icu_adm = []
        self.pid_to_cstatus = dict(zip(t_person.v_pnr_enc,t_person.C_STATUS))
        self.pid_to_dstatus = dict(zip(t_person.v_pnr_enc,t_person.D_STATUS_HEN_START))
        self.pid_to_bdate = dict(zip(t_person.v_pnr_enc,t_person.D_FODDATO))
        self.pid_to_sex = dict(zip(
                                    t_person.v_pnr_enc, 
                                    map(lambda x:float("M" in x), t_person.C_KON)
                                ))

        all_valid_pids = [p for p in self.adm_data if self.adm_data[p]['split_group']==split_group]
        total_number_adm = sum([len(self.adm_data[p]['admissions']) for p in all_valid_pids])
        print (f"Initial number of patient in {self.split_group} is {len(all_valid_pids)}")
        print (f"Initial number of admissions in {self.split_group} is {total_number_adm}")
        
        self.valid_indexes = []
        self.original_label = []
        for p in tqdm(all_valid_pids):
            if self.valid_patient(p):
                icu_idxs = [i for i, a in enumerate(self.adm_data[p]['admissions']) 
                    if a['k_recnum'] in self.adm_data[p]['icu_recnum']]
                icu_adm_left = len(icu_idxs)
                for idx_adm, adm in enumerate(self.adm_data[p]['admissions']):
                    age_at_adm = (adm['adm_datetime'] - parse_date(self.pid_to_bdate[p]))//(31536000) 
                    is_valid_adm = self.valid_adm(p, idx_adm, icu_idxs)
                    if is_valid_adm and age_at_adm > args.minimum_age_at_adm:
                        label, time_to_event = is_valid_adm
                        if label ==1:
                            icu_adm_left -=1 
                        assert time_to_event >= 0
                        max_baseline = min(time_to_event, 30*24) if split_group == 'train' else time_to_event
                        for baseline in range(0, int(max_baseline),args.trigger_time):
                            
                            if args.baseline_hours is not None and baseline!=args.baseline_hours:
                                continue
                            outcome_in_time_horizon = (time_to_event - baseline)<=args.lookahaed

                            if self.args.binary_prediction:
                                t_label = np.float32(label!=2) if outcome_in_time_horizon else 0
                            else:
                                t_label = label if outcome_in_time_horizon else 2

                            self.valid_indexes.append({"pid":p,
                                                        "idx_adm":idx_adm, 
                                                        "label":t_label, 
                                                        "time_to_event":time_to_event,
                                                        "age_at_adm":age_at_adm,
                                                        "sex":self.pid_to_sex[p],
                                                        "baseline":baseline})
                        self.original_label.append((len(self.valid_indexes)-1, label))
                    
                    if age_at_adm <= args.minimum_age_at_adm:
                        self.filter_summary["Age filter"] +=1 
                        if is_valid_adm and is_valid_adm[0] == 1:
                            self.type_icu_adm.pop()
                            self.filter_summary["Age filter ICU"] +=1

                if (icu_adm_left) and age_at_adm >= args.minimum_age_at_adm:
                    self.filter_summary["Missing Concatenation filter ICU"] +=1

        if self.split_group in ["train", "test"]:
            _stats = self.class_stats
        self.stats = stats if stats else _stats 
        self.weights = self.count_risks()
        
        if split_group == 'train':
            if self.args.verbose:
                print (json.dumps(self.stats, sort_keys=True, indent=4))
            self.num_events = args.num_events = 3 if not args.binary_prediction else 1 #TODO remove -- Binary pred is the only support
            self.num_categories = args.num_categories = int(np.max([el["time_to_event"] for el in self.valid_indexes])*1.2)
        else:
            self.num_events = args.num_events
            self.num_categories = args.num_categories
        
        if self.args.verbose:
            num_final_pids = len(set([el['pid'] for el in self.valid_indexes]))
            num_final_adms = len(self.original_label)
            print (f"FINAL number of patient in {self.split_group} is {num_final_pids}")
            print (f"FINAL number of admissions in {self.split_group} is {num_final_adms}")
            for k,v in self.filter_summary.items():
                print("\t{}={}".format(k, v))

        self.datasources = datasources
        self.con=None

    def valid_patient(self, patient):
        if patient not in self.pid_to_cstatus:
            self.filter_summary["Tperson filter"] +=1
            return False
        if self.pid_to_cstatus[patient] in ['1', '90']:
            if self.ehr_data and patient not in self.ehr_data:
                #no ehr record for this patient
                return False
            else:
                return True
        else:
            self.filter_summary["Tperson filter"] +=1
            if self.adm_data[patient]['icu_recnum']: 
                self.filter_summary["Tperson filter ICU"] +=1 
            return False

    def valid_adm(self, patient , idx_adm, icu_idxs):
        # it cannot be itself an icu admission.
        # it has to be longer then args.baseline_hours (num hours of data to feed the model)
        # if it is concatenated to a previos admission we need to take only the first one

        recnum = self.adm_data[patient]['admissions'][idx_adm]['k_recnum']
        adm_datetime = self.adm_data[patient]['admissions'][idx_adm]['adm_datetime']
        
        if recnum in self.adm_data[patient]['icu_recnum']:
            self.filter_summary["Raw ICU admissions"] += 1
            #this is an icu admission
            return False
        
        #if concatenated to the previous one, then discard it (the ICU admission itself is discarded)
        if idx_adm !=0 and \
            self.get_gap_time(patient, idx_adm) < self.args.concatenate_time and \
            (idx_adm - 1) not in icu_idxs:
            self.filter_summary["Concatenated admissions"] += 1
            return False
        
        endpoint = 0
        while True:
            gap = self.get_gap_time(patient, idx_adm + endpoint, wrt='next')
            if (gap is False) or (gap > self.args.concatenate_time):
                break
            endpoint += 1
        
        #get concatenated los and check if higher then baseline
        concatenated_los = ((self.adm_data[patient]['admissions'][idx_adm + endpoint]['adm_datetime'] + \
                            self.adm_data[patient]['admissions'][idx_adm + endpoint]['los']*3600) - \
                            self.adm_data[patient]['admissions'][idx_adm]['adm_datetime'])//3600

        if icu_idxs:
            # check if this adm is the first one linked to the icu admission 
            linked_to_icu = self.is_linked_to_icu(patient, idx_adm, endpoint, icu_idxs)
        else:
            linked_to_icu = False

        if self.adm_data[patient]['in_hospital_death']:
            linked_to_death = self.adm_data[patient]['in_hospital_death'][0]
        else:
            linked_to_death = None
        
        valid_outcome = self.get_outcome(patient, adm_datetime, concatenated_los, linked_to_icu, linked_to_death)

        return valid_outcome
    
    def get_gap_time(self, patient, idx_adm, wrt='prev'):
        if wrt == 'prev':
            wrt_idx = idx_adm -1
            los_idx = idx_adm -1
        elif wrt == 'next':
            wrt_idx = idx_adm + 1
            los_idx = idx_adm
        
        try:
            time_between_adm = abs(self.adm_data[patient]['admissions'][idx_adm]['adm_datetime'] - \
                            self.adm_data[patient]['admissions'][wrt_idx]['adm_datetime'])
        except:
            #this means that idx_adm +1 is out of range list
            return False #it is not concatenated - no gap can be calculated

        los = max(self.adm_data[patient]['admissions'][los_idx]['los'], 0) # if los < 0 it means there the discharged datetime is rounded wrong because missing in the table
        gap_time = (time_between_adm//3600 - los)

        return gap_time # the gap time is in hour


    def is_linked_to_icu(self, patient, idx_adm, endpoint, icu_idxs):
        icu_index = 0
        for icu_i in icu_idxs:
            if idx_adm < icu_i:
                icu_index = icu_i
                break
        
        idx_limit = (idx_adm + endpoint) if self.args.force_icu_concatenation else (idx_adm + endpoint +1)
        
        admtoicu = (self.adm_data[patient]['admissions'][icu_index]['adm_datetime'] - \
                        self.adm_data[patient]['admissions'][idx_adm]['adm_datetime'])//3600
        
        if icu_index > idx_adm and icu_index <= idx_limit:
            self.type_icu_adm.append(self.adm_data[patient]['admissions'][icu_index]['k_recnum'])
            assert type(admtoicu) is float
            return admtoicu
        else:
            return False

    def get_outcome(self, patient, adm_datetime, los, linked_to_icu, linked_to_death):
        # return label and time_to_event
        # 0 in censored
        # 1 is admitted to ICU
        # 2 is discharged
        # 3 is death in the general department
        if type(linked_to_icu) is float:
            #this is out most important outcome department that links with icu
            time_to_event = linked_to_icu
            label = 1 #****patient goes to icu****
        else:

            dstatus = parse_date(self.pid_to_dstatus[patient])
            disch_timestamp = (adm_datetime + los*3600)
            h2dtstaus =  (dstatus - disch_timestamp)/3600

            if not linked_to_death \
                and h2dtstaus//24 <= 14 \
                and self.pid_to_cstatus[patient] == '90':
                self.filter_summary['Death within 14d from discharge'] += 1

            if linked_to_death:
                h2death = (linked_to_death - disch_timestamp)/3600
            else:
                h2death = None

            #check if patient still alive
            if h2dtstaus < 24 and (h2death and h2death<=1):
                # either censored or death
                if self.pid_to_cstatus[patient] == '1' and self.args.time_to_event:
                    time_to_event = (dstatus - adm_datetime)/3600
                    label = 0 #****censored**** 
                    return False

                elif self.pid_to_cstatus[patient] == '90': 
                    time_to_event = np.max([(linked_to_death - adm_datetime)/3600, los, 0])
                    label = 3 #****patient dies****
                else:
                    print("Patient dies but no records of death found ")
                    return False
            else:
                time_to_event = los
                label = 2 #****patient discharged****

        if time_to_event > 365*24 or time_to_event < 0:
            self.filter_summary["Timelimit admissions"] += 1
            if label==1:
                self.type_icu_adm.pop()
                self.filter_summary["Timelimit admissions ICU transfer"]+=1
            return False

        return label, time_to_event # the time to event is in hours

    def count_risks(self):
        dict_weights = defaultdict(lambda :0)
        dict_baselines = defaultdict(lambda :defaultdict(lambda :0))
        for sample in  self.valid_indexes:
            dict_weights[sample["label"]] +=1
            dict_baselines[sample["baseline"]][sample["label"]] +=1

        print ("""Number of samples per label\n""")
        for k,v in dict_weights.items():
            print(f"{k}: {v}\n")
        print ("""Number of samples per baseline\n""")
        dict_baselines = {k:sum(v.values()) for k,v in dict_baselines.items()}
        
        sum_labels = sum(dict_weights.values())
        sum_baselines = sum(dict_baselines.values())

        label_weights = 1 - np.array([dict_weights[k] for k in sorted(dict_weights)])/sum_labels
        if len(dict_baselines)>1:
            dict_baselines = {k: (dict_baselines[k])/sum_baselines for k in sorted(dict_baselines)}
        else:
            dict_baselines = {k:1 for k in dict_baselines}

        weights = []
        for sample in self.valid_indexes:
            label = sample["label"]
            weights.append(label_weights[int(label)] * dict_baselines[sample["baseline"]])
        return weights

    @property
    def class_stats(self):
        stats = {}
        def get_stats_from_indexes(valid_indexes, ft):
            if type(ft) is str:
                items = [el[ft] for el in valid_indexes]
                counter = len(set(items)) <= 2
                stand_fact = {"mean":np.mean(items),
                            "std": np.std(items),
                            "median":np.median(items),
                            "25th":np.quantile(items, 0.25),
                            "75th":np.quantile(items, 0.75)}
                if counter:
                    feat_counter = Counter(items)
                    feat_counter = {str(k):[v, v/sum(feat_counter.values())] for k,v in feat_counter.items()}
                    stand_fact["count"] = feat_counter
                return stand_fact

            elif type(ft) is list:
                return dict(zip(ft,[get_stats_from_indexes(valid_indexes, el)for el in ft]))

        indexes = [self.valid_indexes[i] for i,_ in self.original_label]
        stats["age_at_adm"] = get_stats_from_indexes(indexes, "age_at_adm")
        stats["idx_adm"] = get_stats_from_indexes(indexes, "idx_adm")
        stats["sex"] = get_stats_from_indexes(indexes, "sex")
        
        if self.args.verbose:
            stats["discharge"] = get_stats_from_indexes(
                [self.valid_indexes[i] for i,label in self.original_label if label==2], 
                ['age_at_adm', "idx_adm", "sex", "time_to_event"])
            stats["icu_admissions"] = get_stats_from_indexes(
                [self.valid_indexes[i] for i,label in self.original_label if label==1], 
                ['age_at_adm', "idx_adm", "sex", "time_to_event"])
            stats["death"] = get_stats_from_indexes(
                [self.valid_indexes[i] for i,label in self.original_label if label==3], 
                ['age_at_adm', "idx_adm", "sex", "time_to_event"])

            stats["patients"] = len(set([el['pid'] for el in self.valid_indexes]))
            stats["admissions"] = len(self.original_label)
            stats["binary_assessments"] = get_stats_from_indexes(self.valid_indexes, "label")
            stats["baselines"] = get_stats_from_indexes(self.valid_indexes, "baseline")

            icu_info = pd.read_csv(open(os.path.join(self.args.input_dir, "icu_df.tsv"), 'r'), sep='\t')
            icu_info = icu_info.drop_duplicates(subset=['recnum'])
            recnums = np.array(self.type_icu_adm)
            recnums2surgical = dict(zip(icu_info.recnum, icu_info.surgical))
            count_type_icu_adm = Counter([recnums2surgical[r] for r in recnums])
            count_type_icu_adm = {k:[v, v/sum(count_type_icu_adm.values())] for k,v in count_type_icu_adm.items()}
            stats["type_icu_adm"] = count_type_icu_adm

            count_outcome = Counter([l for _,l in self.original_label])
            count_outcome = {k:[v, v/sum(count_outcome.values())] for k,v in count_outcome.items()}
            stats['outcome'] = count_outcome
            stats_path = os.path.join(self.args.input_dir, "stats_{}.json".format(self.split_group))
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
        return stats

    def __len__(self):
        return len(self.valid_indexes)
            
    def __getitem__(self, index):
        pass
