import psycopg2
import json
import argparse
from pathlib import Path
import sys 
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from unplanned_net.utilities.date import timestamp_to_datetime
from tqdm import tqdm
import io

class PseudoFile(io.TextIOBase):
    """ given an iterator which yields strings,
    return a file like object for reading those strings """

    def __init__(self, it):
        self._it = it
        self._f = io.StringIO()

    def read(self, length=sys.maxsize):

        try:
            while self._f.tell() < length:
                self._f.write(next(self._it) + "\n")

        except StopIteration as e:
            pass

        finally:
            self._f.seek(0)
            data = self._f.read(length)

            # save the remainder for next read
            remainder = self._f.read()
            self._f.seek(0)
            self._f.truncate(0)
            self._f.write(remainder)
            return data

    def readline(self):
        return next(self._it)

parser = argparse.ArgumentParser(description='move to postgres database')
parser.add_argument('--data_file', type=str, default="input/metadata_diag_biochem.json", help='data file')
args = parser.parse_args()

ehr = json.load(open(args.data_file, 'r'))
print ("EHR json loaded! ")
pids = list(ehr.keys())
conn = psycopg2.connect("host=trans-db-01 dbname=daplaci user=daplaci")
cur = conn.cursor()


cur.execute("DROP TABLE IF EXISTS jsontable;")
conn.commit()

create_table_query = """CREATE TABLE jsontable (pid text,datetime timestamp,ts double precision,data jsonb);"""
print (create_table_query)
cur.execute(create_table_query)
conn.commit()
conn.close()

conn = psycopg2.connect("host=trans-db-01 dbname=daplaci user=daplaci")
def row_generator():
    for pt in tqdm(pids):
        for dt in ehr[pt]:
            pid_to_sql = pt
            datetime_to_sql = str(timestamp_to_datetime(float(dt)))
            row = '\t'.join((pid_to_sql, datetime_to_sql , dt, json.dumps(ehr[pt][dt], ensure_ascii=False)))
            yield row

results = (row for row in row_generator())
output = PseudoFile(results)
with conn.cursor() as cur2:
    cur2.copy_from(output, "jsontable" , sep="\t")

conn.commit()
cur = conn.cursor()
cur.execute("CREATE INDEX jsontable_pid_index ON jsontable (pid);")
conn.commit()

conn.close()

Path(".snakemake_log/create_table.log").touch()
