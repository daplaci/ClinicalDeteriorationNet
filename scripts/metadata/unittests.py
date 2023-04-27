import sys
import os
import unittest
from ast import literal_eval
import numpy as np
import psycopg2
import json
import pandas as pd
import random
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from utilities.date import parse_date, START_BTH, END_BTH

class DataValidityTest(unittest.TestCase):
    """
    test the essential functionality
    """
    def __init__(self, *args, **kwargs):
        super(DataValidityTest, self).__init__(*args, **kwargs)
        self.db_conn = psycopg2.connect("host=trans-db-01 dbname=daplaci user=daplaci")
        self.bth_conn = psycopg2.connect("host=trans-db-01 dbname=bth user=daplaci")
        adm_file = 'input/metadata_admissions.json'
        self.adm_json = json.load(open(adm_file, 'r'))
        self.pids = list(self.adm_json.keys())

    def test_admission(self):
        """
        test admission data
        """
        pid_to_test = 100
        while pid_to_test > 0:
            ##do something
            p = random.choice(self.pids)
            df = pd.read_sql(f"select pid, adm_ymd, adm_hm from epj_dev.adm where pid={p} and konttype='I' and admway='A'", self.bth_conn)
            timestamps = map(parse_date, df.adm_ymd+df.adm_hm)
            for t in timestamps:
                if t>END_BTH or t<START_BTH:                
                    continue
                self.assertTrue(t in [a['adm_datetime'] for a in self.adm_json[p]['admissions']])
            pid_to_test-=1

    def test_diagnoses(self):
        """
        test diagnoses data
        """
        pid_to_test = 100
        while pid_to_test > 0:
            ##do something
            p = random.choice(self.pids)
            df = pd.read_sql(f"select pid, ts, datetime, data->'diag' as diag from jsontable where pid='{p}';", self.db_conn)
            if len(df) == 0:
                continue
            lpr2bth_df = pd.read_sql(f"""select tb1.departure_date, tb1.departure_datetime, tb2.diag_code from 
(select person_id, visit_id, departure_date, departure_datetime from lpr_210312.adm where person_id={p} and arrival_date<'2018-04-01') as tb1
inner join (select visit_id, diag_code from lpr_210312.diag) as tb2
on tb1.visit_id=tb2.visit_id""", self.bth_conn)
            
            #check date
            for date in df[~df.diag.isna()].datetime:
                if not date.date() in lpr2bth_df.departure_date.tolist():
                    import pdb;pdb.set_trace
                self.assertTrue(date.date() in lpr2bth_df.departure_date.tolist())
            #if time is available it has to match
            #self.assertTrue(lpr2bth_df.dropna().departure_datetime.isin(df[~df.diag.isna()].datetime).all()) TEMPORARY BUG departure date and departure datetime mismatch
            for dt, diag in zip(df.datetime, df.diag):
                if not diag: continue
                for d in diag:
                    self.assertTrue(d in lpr2bth_df[lpr2bth_df.departure_date == dt.date()].diag_code.tolist())
            pid_to_test-=1
        
### Run the tests
if __name__ == '__main__':
    unittest.main()
