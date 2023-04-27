import json
import sys,os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from unplanned_net.utilities.date import *
from collections import defaultdict
from tqdm import tqdm
from scipy import stats
import pandas as pd
import pickle
import argparse
import pandas as pd

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

parser = argparse.ArgumentParser(description='add biochem to diag metadata')
parser.add_argument('--t_diag_file', type=str, help='file containing the admission dataset')
parser.add_argument('--t_adm_file', type=str, help='file containing the admission dataset')
parser.add_argument('--metadata_admissions_file', type=str, help='metadata_admissions')
parser.add_argument('--metadata_diag_file', type=str, help='metadata_admissions')
parser.add_argument('--labka_file', type=str, 
                    help='file containing the admission dataset')
parser.add_argument('--metadata_diag_biochem_file', type=str, 
                    help='metadata_admissions')

args = parser.parse_args()


with open(args.metadata_diag_file, 'r') as f:
    m = json.load(f)

with open(args.metadata_admissions_file, 'r') as f:
    admissions = json.load(f)


f = open(args.labka_file, 'r')
cols = next(f).strip().split("\t")
pbar = tqdm(total=400929234)

for i,line in enumerate(f):
    pbar.update(1)
    line = line.split('\t')
    l = dict(zip(cols, line))
    try:
        admissions[l["pid"]]
    except:
        continue
    
    quantity_id = "{}_{}_{}".format(l["component_simple_lookup"], l["system_clean"], l["unit_clean"])
    quantity_id = quantity_id.replace(' ', '-')

    if l["pid"] not in m:
        m[l["pid"]] = {}

    try:
        date_record = str(parse_date("{}T{}Z".format(l["date"],l["time"]) )) #is_utc=(l["database"]=='labka')
    except Exception as e:
        print (e)
        
    if date_record not in m[l["pid"]]:
        m[l["pid"]][date_record] = {}

    if is_number(l["shown_clean"]):
        current_lab_value = (float(l["shown_clean"]))
    else:
        current_lab_value = l["shown_clean"].replace(' ', '-')

    biochem = '{}@{}'.format(quantity_id, current_lab_value)
    
    if 'biochem' in m[l["pid"]][date_record]:
        m[l["pid"]][date_record]['biochem'].append(biochem)
    else:
        m[l["pid"]][date_record].update({'biochem':[biochem]})

with open(args.metadata_diag_biochem_file, 'w') as f:
    json.dump(m, f)


