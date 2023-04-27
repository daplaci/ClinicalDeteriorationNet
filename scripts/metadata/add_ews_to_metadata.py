import json
import sys,os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from collections import defaultdict
from tqdm import tqdm
import argparse
from unplanned_net.utilities.date import parse_date
from unplanned_net.utilities.time import timeit

parser = argparse.ArgumentParser(description='add ews to file')
parser.add_argument('--ews_file', type=str, help='file containing the ews records')
parser.add_argument('--metadata_admissions_tmp', type=str, default="input/metadata_admissions_tmp.json", help='metadata_admissions')
parser.add_argument('--metadata_admissions_file', type=str, default="input/metadata_admissions.json", help='metadata_admissions')

args = parser.parse_args()

dict_ews = defaultdict(dict)
f = open(args.ews_file, 'r')
header_ews = next(f).strip().split('\t')

for l in f:
    line = l.strip().split('\t')
    line_dict = dict(zip(header_ews, line))
    dict_ews[line_dict["PID"]][parse_date(line_dict["REKVDT"])] = float(line_dict["VALUE"])

metadata_admissions = json.load(open(args.metadata_admissions_tmp, 'r'))

@timeit()
def add_ews(metadata_admissions, dict_ews):
    for p, d in tqdm(metadata_admissions.items()):
        metadata_admissions[p]["ews"] = defaultdict(dict)
        if p in dict_ews:
            for ts, value in dict_ews[p].items(): 
                metadata_admissions[p]["ews"][ts] = value

add_ews(metadata_admissions, dict_ews)

with open(args.metadata_admissions_file, 'w') as f:
    json.dump(metadata_admissions, f, ensure_ascii=False)