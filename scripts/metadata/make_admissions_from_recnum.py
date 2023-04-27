import pandas as pd
import json
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from unplanned_net.utilities.date import parse_date, START_BTH, END_BTH, get_bth_date_adm, get_bth_date_disch
from tqdm import tqdm
from collections import defaultdict
import argparse
import pickle

parser = argparse.ArgumentParser(description='make_recnum_to_sksube')
parser.add_argument('--t_adm_file', type=str, help='file containing the admission dataset')
parser.add_argument('--recnum_to_ube_file',type=str, help='file recnum_to_ube_file')
parser.add_argument('--metadata_admissions_file', type=str, help='metadata_admissions')

args = parser.parse_args()

print ("Loading files..\n")
mapper = pd.read_csv('icu_mapper_new.tsv', sep=';').astype(str)
combined_code2surgical = dict(zip(mapper.COMBINED_CODE, mapper.SURGICAL))
only_icu_mapper = mapper[(mapper.INCLUDE=='1')][['COMBINED_CODE']]
list_icu_department = only_icu_mapper.COMBINED_CODE.tolist()
print ("ICU department: ", list_icu_department)

print ("\nLoading recnum to ube {}:".format(args.recnum_to_ube_file))
with open(args.recnum_to_ube_file, 'rb') as f:
    #this dictionary contains for each recnum in t_sksube all the ube codes related to it
    recnum_to_ube = pickle.load(f)


print ("\nGenerating metadata_admissions.. \n")
metadata_admissions = defaultdict(lambda : {'admissions':[], 'icu_recnum':[], 'in_hospital_death':[]})
pbar = tqdm(total=21539834)
f = open(args.t_adm_file, 'r')
cols = next(f).split("\t")

df_icu_admissions = [] #list of tuple to fill the dataframe

for i,l in enumerate(f):
    pbar.update(1)
    line_dict = dict(zip(cols, l.split("\t")))

    #here all the filtering
    #take only inpatients
    if line_dict['KONTTYPE']!='I':
        continue

    adm_datetime = parse_date(get_bth_date_adm(line_dict))
    disch_date = parse_date(get_bth_date_disch(line_dict))

    #filter inside DATA period
    if  (not adm_datetime or 
        not disch_date or 
        adm_datetime > END_BTH or 
        adm_datetime < START_BTH):
        continue

    los = ((disch_date) - (adm_datetime))//3600
    combined_code = "{}{}".format(line_dict["HOSPID"], line_dict["WARDID"].replace(".", ""))

    # here the criterion for assessing if an admission is an ICU admission
    is_icu_dept = combined_code in list_icu_department
    is_sks_icu = (line_dict['RECNUM'] in recnum_to_ube)
    is_unplanned = line_dict["ADMWAY"] == "A"
    
    if is_icu_dept and is_sks_icu:
        df_icu_admissions.append((line_dict["RECNUM"], line_dict["ADM_YMD"], line_dict["ADMWAY"], combined_code, combined_code2surgical[combined_code]))
        if is_unplanned: 
            #this means the icu admission is unplanned
            metadata_admissions[line_dict['PID']]["icu_recnum"].append(line_dict["RECNUM"])
        else:
            continue

    in_hospital_death = line_dict["ENDWAY"] =='DØ'
    if in_hospital_death:
        metadata_admissions[line_dict['PID']]["in_hospital_death"].append(disch_date)
    
    adm_dict = {"k_recnum": line_dict["RECNUM"], "adm_datetime": adm_datetime, "dept": combined_code, 'los':los}
    metadata_admissions[line_dict['PID']]["admissions"].append(adm_dict)

for pid in tqdm(metadata_admissions):
    metadata_admissions[pid]['admissions'] = sorted(metadata_admissions[pid]['admissions'], key=lambda k: k['adm_datetime'])
    metadata_admissions[pid]['icu_recnum'] = [adm['k_recnum'] for adm in metadata_admissions[pid]['admissions'] if adm['k_recnum'] in metadata_admissions[pid]['icu_recnum']]

print ("\nICU stats :")
icu_df = pd.DataFrame.from_records(df_icu_admissions, columns=["recnum", "admdate", "planned", "combined_code", "surgical"])
icu_df.to_csv("input/icu_df.tsv", sep='\t', index=False)

print ("\nSaving metadata file..{}".format(args.metadata_admissions_file))
with open(args.metadata_admissions_file, 'w') as f:
    json.dump(metadata_admissions, f, ensure_ascii=False)

print ("File Admissions Saved\n\n\n")

        
