import json
import argparse
import pandas as pd
import numpy as np

    
parser = argparse.ArgumentParser(description='ICU Survival model')
parser.add_argument('--metadata_admissions_file', type=str, default='input/metadata_admissions.json', help='file containing the admission dataset')
parser.add_argument('--test_size', type=float, default=0.2, help='percentage of test to held out')
parser.add_argument('--seed', type=int, default=42, help='seed for random numpy')

#args for notes
args = parser.parse_args()

np.random.seed(args.seed)

adm_data = json.load(open(args.metadata_admissions_file,'r'))
for p in sorted(adm_data):
    adm_data[p]['split_group'] = np.random.choice(['train','test'], p=[(1-args.test_size),args.test_size])

with open(args.metadata_admissions_file.replace(".json", "_train.json"), 'w') as f:
    d = {k:v for k,v in adm_data.items() if adm_data[k]['split_group'] == 'train'}
    json.dump(d, f)

with open(args.metadata_admissions_file.replace(".json", "_test.json"), 'w') as f:
    d = {k:v for k,v in adm_data.items() if adm_data[k]['split_group'] == 'test'}
    json.dump(d, f)
