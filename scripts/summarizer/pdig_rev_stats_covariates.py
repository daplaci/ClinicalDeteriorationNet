import json
import sys
assert sys.version_info > (3,5)
import shlex
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pickle
from collections import defaultdict, Counter
PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
sys.path.append(PATH_TO_LIB)
import unplanned_net.utilities.parser as parser
from unplanned_net.dataset.datasource import get_datasources, TextDataSource, NumericalDataSource
from unplanned_net.dataset.notes_datasource import NotesDataSource
import unplanned_net.dataset.data_utils as data_utils
import unplanned_net.run.model_utils as model_utils
import unplanned_net.run.learn as learn
from unplanned_net.utilities.html_writer import HtmlWriter
from unplanned_net.utilities.calibration import IsotonicLayer, CalibratedModel
from unplanned_net.utilities.vocab import *                                                     
from unplanned_net.utilities.collections import get_best_exp_optuna
from unplanned_net.utilities.visualization import plot_optuna, plot_attributions, plot_calibration, plot_catches


collect_args = argparse.ArgumentParser(description='parse args for performance analysis')
collect_args.add_argument('--moab_path', 
    type=str,
    required=True,
    help='path where to run this collection - if none use the running directory')
collect_args = collect_args.parse_args()


if __name__ == "__main__":
    if collect_args.moab_path:
        print (f"Collecting studies in {collect_args.moab_path}")
        os.chdir(collect_args.moab_path)

    icd_mapper_name = ""#path to mapper ICD -> definition
    icd_mapper = pd.read_csv(icd_mapper_name, sep='\t', 
                        names = ['icd10', 'definition', 'chapter_name', 'chapter', 'block_name', 'block'])
    icd2definition = dict(zip(icd_mapper.icd10, icd_mapper.definition))
    icd2definition['R67'] = "Findings in assessing general functional ability"
    
    auctable = pd.read_csv('AUC_history_gridsearch.tsv', sep='\t')

    studies = set(auctable.study_id.tolist())
    if len(studies) > 1:
        raise NotImplementedError ("Only one study per grid search is currently implented in the evaluation")
    try:
        best_trials = get_best_exp_optuna(studies)
        best_exp_id = best_trials[0]#auctable.iptloc[best_trials[0].number - 1 ].exp_id
        for s in studies:
            plot_optuna(s)
    except Exception as e:
        print (f"{e}.\nError: could not load the study from the optnua storage.. reading it from the auctable")
        best_exp_id = auctable[auctable.val_auprc == max(auctable.val_auprc)].exp_id.tolist()[0]
    best_exp_id = [best_exp_id]

    for exp_id in best_exp_id:
        print ("Best exp id {}".format(exp_id))
        moab_filename = [f for f in os.listdir("moab_jobs/") if exp_id in f][0]
        flag_string = open("moab_jobs/{}".format(moab_filename), 'r').readlines()[-1]
        args = parser.parse_args(shlex.split(flag_string)[3:])
        args.num_workers = 0
        args.train_batch = 16
        args.val_batch = 16
        args.verbose = False

        datasources = get_datasources(args)
        adm_data, ehr_data, t_person = data_utils.get_input_data(args)
        adm_test = json.load(open(os.path.join(args.input_dir, args.admissions_test),'r'))
                
        for p in sorted(adm_data):
            adm_data[p]['split_group'] = np.random.choice(['train','val'], p=[1.0, 0.0])

        train_dataset, train_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 'train', args, 
                                datasources=datasources, 
                                use_weights_sampler=False)

        for p in adm_test:
            adm_test[p]['split_group'] = 'test'

        val_dataset, val_loader = data_utils.get_dataclasses(adm_test, ehr_data, t_person, 'test', args, 
                                datasources=datasources, 
                                use_weights_sampler=False, 
                                stats=train_dataset.stats)

        for ds in datasources:
            if isinstance(ds, NumericalDataSource):
                ds.set_stats(train_dataset.stats)
        
        print ("Running Training stats on test.. \n")
        for split in [train_dataset, val_dataset]:
            val_pid_adm_tte = [
                (   
                    i, 
                    split.valid_indexes[i]['pid'],
                    split.valid_indexes[i]['idx_adm'], 
                    split.valid_indexes[i]['time_to_event']
                
                )  for i,_ in split.original_label
            ]

            comorbidities = Counter()
            stats_data = {
                'dept':[],
                'comorbidities':[], 
                'biochem_24':[], 
                'biochem_48':[], 
                'biochem_72':[], 
                'biochem_tte':[],
                'notes_24':[], 
                'notes_48':[], 
                'notes_72':[], 
                'notes_tte':[],
                }

            for i, pid, adm, tte in tqdm(val_pid_adm_tte):
                
                stats_data['dept'].append(split.adm_data[pid]['admissions'][adm]['dept'])
                diag_text = set(split.__get_data_for_stats__(pid, adm, baseline=0, datasource='diag'))
                comorbidities.update(diag_text)
                stats_data['comorbidities'].append(len(diag_text))
                if args.biochem and args.notes:
                    if tte > 24:
                        stats_data['biochem_24'].append(len(split.__get_data_for_stats__(pid, adm, baseline=24, datasource='biochem')))
                        stats_data['notes_24'].append(len(split.__get_data_for_stats__(pid, adm, baseline=24, datasource='notes')))
                    else:
                        stats_data['biochem_24'].append(np.nan)
                        stats_data['notes_24'].append(np.nan)  
                    
                    if tte > 48:
                        stats_data['biochem_48'].append(len(split.__get_data_for_stats__(pid, adm, baseline=48, datasource='biochem')))
                        stats_data['notes_48'].append(len(split.__get_data_for_stats__(pid, adm, baseline=48, datasource='notes')))
                    else:
                        stats_data['biochem_48'].append(np.nan)
                        stats_data['notes_48'].append(np.nan)

                    if tte > 72:
                        stats_data['biochem_72'].append(len(split.__get_data_for_stats__(pid, adm, baseline=72, datasource='biochem')))
                        stats_data['notes_72'].append(len(split.__get_data_for_stats__(pid, adm, baseline=72, datasource='notes')))
                    else:
                        stats_data['biochem_72'].append(np.nan)
                        stats_data['notes_72'].append(np.nan)

                    stats_data['biochem_tte'].append(len(split.__get_data_for_stats__(pid, adm, baseline=tte, datasource='biochem')))
                    stats_data['notes_tte'].append(len(split.__get_data_for_stats__(pid, adm, baseline=tte, datasource='notes')))
            
            df = pd.DataFrame(stats_data)
            dept_classification = pd.read_csv('shakcomplete_2020-01-07.tsv', sep='\t')
            dept2type = dict(zip(dept_classification.SHAK, dept_classification.A_GROUP))
            df['dept_type'] = df.dept.apply(lambda x : dept2type[x])
            df.to_csv("clinical_deterioration/figures/pdig_revision/{}_{}_stats.csv".format(split.split_group, exp_id), index=False)

            comorb_df = pd.DataFrame.from_dict(comorbidities, orient='index', columns=['count']).reset_index().rename(columns={'index':'icd'})
            comorb_df['icd_level3'] = comorb_df.icd.apply(lambda x : x[1:4] if x[0] == 'D' else x[:3])
            ordered_comorb_df = comorb_df.groupby('icd_level3')['count'].sum().sort_values(ascending=False).reset_index()
            ordered_comorb_df.to_csv("clinical_deterioration/figures/pdig_revision/{}_{}_grouped_comorbidities.csv".format(split.split_group, exp_id), index=False)

            with open('clinical_deterioration/figures/pdig_revision/{}_{}_comorbidities.pkl'.format(split.split_group, exp_id), 'wb') as f:
                pickle.dump(comorbidities, f)


stats_path = "clinical_deterioration/figures/pdig_revision/"
train_stats = pd.read_csv(os.path.join(stats_path, "train_8ffc57282cb33791c148ba0316c03f8a_stats.csv"))
test_stats = pd.read_csv(os.path.join(stats_path, "test_8ffc57282cb33791c148ba0316c03f8a_stats.csv"))
train_dept_counter = Counter(train_stats["dept_type"])
test_dept_counter = Counter(test_stats["dept_type"])

train_comorb = pd.read_csv(os.path.join(stats_path, "train_8ffc57282cb33791c148ba0316c03f8a_grouped_comorbidities.csv"))
test_comorb = pd.read_csv(os.path.join(stats_path, "test_8ffc57282cb33791c148ba0316c03f8a_grouped_comorbidities.csv"))

print("##### TRAIN STATS #####")
print(f""" Previous Diagnosis --> "median":{np.nanmedian(train_stats.comorbidities)}, 
"25th":{np.nanquantile(train_stats.comorbidities, 0.25)},
"75th":{np.nanquantile(train_stats.comorbidities, 0.75)}""")

print(f""" biochem@24 --> "median":{np.nanmedian(train_stats.biochem_24)}, 
"25th":{np.nanquantile(train_stats.biochem_24, 0.25)},
"75th":{np.nanquantile(train_stats.biochem_24, 0.75)}""")
print(f"""biochem@48 --> "median":{np.nanmedian(train_stats.biochem_48)}, 
"25th":{np.nanquantile(train_stats.biochem_48, 0.25)},
"75th":{np.nanquantile(train_stats.biochem_72, 0.75)}""")
print(f"""biochem@72 --> "median":{np.nanmedian(train_stats.biochem_72)}, 
"25th":{np.nanquantile(train_stats.biochem_72, 0.25)},
"75th":{np.nanquantile(train_stats.biochem_72, 0.75)}""")

print(f"""notes@24 --> "median":{np.nanmedian(train_stats.notes_24)},
"25th":{np.nanquantile(train_stats.notes_24, 0.25)},
"75th":{np.nanquantile(train_stats.notes_24, 0.75)}""")
print(f"""notes@48 --> "median":{np.nanmedian(train_stats.notes_48)},
"25th":{np.nanquantile(train_stats.notes_48, 0.25)},
"75th":{np.nanquantile(train_stats.notes_48, 0.75)}""")
print(f"""notes@72 --> "median":{np.nanmedian(train_stats.notes_72)},
"25th":{np.nanquantile(train_stats.notes_72, 0.25)},
"75th":{np.nanquantile(train_stats.notes_72, 0.75)}""")

print("##### TEST STATS #####")

print(f""" Previous Diagnosis --> "median":{np.nanmedian(test_stats.comorbidities)},
"25th":{np.nanquantile(test_stats.comorbidities, 0.25)},
"75th":{np.nanquantile(test_stats.comorbidities, 0.75)}""")

print(f""" biochem@24 --> "median":{np.nanmedian(test_stats.biochem_24)},
"25th":{np.nanquantile(test_stats.biochem_24, 0.25)},
"75th":{np.nanquantile(test_stats.biochem_24, 0.75)}""")
print(f"""biochem@48 --> "median":{np.nanmedian(test_stats.biochem_48)},
"25th":{np.nanquantile(test_stats.biochem_48, 0.25)},
"75th":{np.nanquantile(test_stats.biochem_72, 0.75)}""")
print(f"""biochem@72 --> "median":{np.nanmedian(test_stats.biochem_72)},
"25th":{np.nanquantile(test_stats.biochem_72, 0.25)},
"75th":{np.nanquantile(test_stats.biochem_72, 0.75)}""")

print(f"""notes@24 --> "median":{np.nanmedian(test_stats.notes_24)},
"25th":{np.nanquantile(test_stats.notes_24, 0.25)},
"75th":{np.nanquantile(test_stats.notes_24, 0.75)}""")
print(f"""notes@48 --> "median":{np.nanmedian(test_stats.notes_48)},
"25th":{np.nanquantile(test_stats.notes_48, 0.25)},
"75th":{np.nanquantile(test_stats.notes_48, 0.75)}""")
print(f"""notes@72 --> "median":{np.nanmedian(test_stats.notes_72)},
"25th":{np.nanquantile(test_stats.notes_72, 0.25)},
"75th":{np.nanquantile(test_stats.notes_72, 0.75)}""")

print(f"""Top 5 most commont department in train: 
{[(k,v,v/len(train_stats)) for k,v in train_dept_counter.most_common(5)]}
"""
)

print(f"""Top 5 most commont department in test:
{[(k,v,v/len(test_stats)) for k,v in test_dept_counter.most_common(5)]}
""")

train_comorb_subset = train_comorb[~train_comorb.icd_level3.str.startswith(('Z', 'S'))].iloc[:10]
print(f"""Top 10 most commont diagnoses in train: 
{[(icd, tot, tot/len(train_stats)) for icd, tot in train_comorb_subset.itertuples(index=False)]}
"""
)

test_comorb_subset = test_comorb[~test_comorb.icd_level3.str.startswith(('Z', 'S'))].iloc[:10]
print(f"""Top 10 most commont diagnoses in test:
{[(icd, tot, tot/len(test_stats)) for icd, tot in test_comorb_subset.itertuples(index=False)]}
""")

assert sum(train_dept_counter.values()) == len(train_stats)
assert sum(test_dept_counter.values()) == len(test_stats)
                