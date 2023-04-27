import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import json
from tqdm import tqdm
from collections import defaultdict
from unplanned_net.utilities.date import *
import argparse

parser = argparse.ArgumentParser(description='make_recnum_to_sksube')
parser.add_argument('--t_diag_file', type=str, help='file containing the admission dataset')
parser.add_argument('--t_adm_file', type=str, help='file containing the admission dataset')
parser.add_argument('--metadata_admissions_file', type=str, help='metadata_admissions')
parser.add_argument('--metadata_diag_file', type=str, help='metadata_admissions')

args = parser.parse_args()

print ("Collecting diags.. \n")

recnum_to_diag = defaultdict(list)
f = open(args.t_diag_file, 'r')
pbar = tqdm(total=89517301)
for i,l in enumerate(f):
    pbar.update(1)
    if i == 0:
        cols = l.split("\t")
        continue
    line_dict = dict(zip(cols, l.split("\t")))

    recnum_to_diag[line_dict['V_RECNUM']].append(line_dict["C_DIAG"])
    

print ("Matching metadata t_diag with t_adm.. \n")

with open(args.metadata_admissions_file, 'r') as f:
    admissions = json.load(f)

input_diag = defaultdict(dict)
pbar = tqdm(total=56606125)
f = open(args.t_adm_file, 'r')
cols = next(f).split("\t")
for i,l in enumerate(f):
    pbar.update(1)
    line_dict = dict(zip(cols, l.split("\t")))

    try:
        #if no admission record is found for this path, it has no sense to create an input data for this patient
        admissions[line_dict["v_cpr_enc"]]
    except:
        continue

    adm_datetime = get_date_disch(line_dict)
    timestamp = parse_date(adm_datetime)

    assert line_dict["k_recnum"] not in input_diag[line_dict["v_cpr_enc"]]

    input_diag[line_dict["v_cpr_enc"]][timestamp] = {'diag':recnum_to_diag[line_dict['k_recnum']]}   

print ("Saving diags..")
with open(args.metadata_diag_file, 'w') as f:
    json.dump(input_diag, f)
        