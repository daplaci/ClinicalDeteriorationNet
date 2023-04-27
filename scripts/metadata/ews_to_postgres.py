import psycopg2
import argparse
import sys 
from pathlib import Path
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from unplanned_net.utilities.date import parse_date, START_BTH, END_BTH, timestamp_to_datetime

parser = argparse.ArgumentParser(description='move to postgres database')
parser.add_argument('--data_file', type=str, help='data file')
args = parser.parse_args()

ewstable = open(args.data_file, 'r')
cols = next(ewstable).strip().split("\t")
print ("EHR json loaded! ")
conn = psycopg2.connect("host=trans-db-01 dbname=daplaci user=daplaci")
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS ewstable;")
conn.commit()


create_table_query = """CREATE TABLE ewstable (pid text,datetime timestamp,ts double precision,ews text);"""
print (create_table_query)
cur.execute(create_table_query)
conn.commit()

for i,l in enumerate(ewstable):
    line_dict = dict(zip(cols, l.split("\t")))
    source_to_sql = line_dict["VALUE"]
    pid_to_sql = line_dict["PID"]
    ts_to_sql = parse_date(line_dict["REKVDT"])
    datetime_to_sql = timestamp_to_datetime(ts_to_sql)
    insert_statement = "INSERT INTO notestable VALUES ('{}','{}',{},'{}');".format(pid_to_sql, datetime_to_sql , ts_to_sql, source_to_sql) 
    cur.execute(insert_statement)   


cur.execute("CREATE INDEX ewstable_pid_index ON ewstable (pid);")
conn.commit()
conn.close()

Path("input/.sql_ews").touch()