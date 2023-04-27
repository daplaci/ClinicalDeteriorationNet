import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from unplanned_net.utilities.date import *
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

parser = argparse.ArgumentParser(description='make_recnum_to_sksube')

parser.add_argument('--labka_file', type=str, 
    help='file containing the admission dataset')

parser.add_argument('--biochem_values_file', type=str, 
    default="input/biochem_values.pkl", help='file recnum_to_ube_file')

parser.add_argument('--biochem_top_file', type=str, 
    default="input/biochem_top.pkl", help='file recnum_to_ube_file')


args = parser.parse_args()

f = open(args.labka_file, 'r')
cols = next(f).split("\t")

biochem_set = defaultdict(list)
biochem_count = defaultdict(lambda : 0)
biochem_top = defaultdict(list)


for i,line in enumerate(f):
    line = line.split('\t')
    l = dict(zip(cols, line))

    quantity_id = "{}_{}_{}".format(l["component_simple_lookup"], l["system_clean"], l["unit_clean"])
    quantity_id = quantity_id.replace(' ', '-')
    biochem_count[quantity_id]+=1

    if is_number(l["shown_clean"]):
        biochem_set[quantity_id].append(float(l["shown_clean"]))
    else:
        biochem_set[quantity_id].append(l["shown_clean"].replace(' ', '-'))

print ("generating qcut")
biochem_values = dict()
for k,v in biochem_set.items():

    v = list(v)
    float_v = [el for el in v if is_number(el)]
    text_v = [el for el in v if not is_number(el)]
    
    biochem_values[k] = {}
    if float_v:
        for bin_ in [10,50,100]:
            if len(v) > 1:
                _, biochem_values[k][bin_] = pd.qcut(float_v, bin_, retbins=True, duplicates="drop")
            else:
                biochem_values[k][bin_] = np.array(float_v)
    if text_v:
        biochem_values[k]['text'] = np.array(text_v)

for k,v in biochem_count.items():
    biochem_top[v].append(k)

biochem_popularity_sorted = [b for k in sorted(list(biochem_top.keys()), reverse=True) for b in sorted(biochem_top[k])]

biochem_topk = dict(zip(biochem_popularity_sorted, range(len(biochem_popularity_sorted))))

with open(args.biochem_top_file, 'wb') as f:
    pickle.dump(biochem_topk, f)

with open(args.biochem_values_file, 'wb') as f:
    pickle.dump(biochem_values, f)
