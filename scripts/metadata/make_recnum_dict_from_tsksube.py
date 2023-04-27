import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from unplanned_net.utilities.date import *
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import argparse
import pickle

parser = argparse.ArgumentParser(description='make_recnum_to_sksube')
parser.add_argument('--input_file', type=str, help='file containing the sks codes for icu department')
parser.add_argument('--recnum_to_ube_file',type=str, default="input/tmp/recnum_to_ube.pkl", help='file recnum_to_ube_file')

args = parser.parse_args()

pbar = tqdm(total=125185443)
recnum_to_ube = defaultdict(set)

f = open(args.input_file, 'r', errors='replace')
cols = next(f).strip().split("\t")

for i,l in enumerate(f):
    pbar.update(1)
    
    line_dict = dict(zip(cols, l.split("\t")))

    recnum_to_ube[line_dict["RECNUM"]].add(line_dict["PROCCODE"])

recnum_with_icu_proc = set()
for key, value in recnum_to_ube.items():
    if any([(v[:4] == "NABE" or v[:4] == "NABB") for v in value]):
        recnum_with_icu_proc.add(key)

print ("\nTotal number of recnum with an icu procedure from 2009 to 2016 is {}".format(len(recnum_with_icu_proc)))

print ("\nSaving recnum to ube..")
with open(args.recnum_to_ube_file, 'wb') as f:
    pickle.dump(recnum_with_icu_proc, f)
