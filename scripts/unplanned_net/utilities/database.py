import psycopg2

class MyDB(object):

    def __init__(self):
        self._db_connection = psycopg2.connect("host=trans-db-01 dbname=daplaci user=daplaci")
        self._db_cur = self._db_connection.cursor()
        self.string_query =  """select pid, ts, jsonb_merge(tb1.data, tb2.data) from 
(SELECT pid, ts, data FROM jsontable where pid=%s) as tb1 
full outer join 
(SELECT pid, ts, data FROM notestable where pid=%s) as tb2 
using (pid, ts) where ts <= %s"""

    def query(self, query, *variables, fetch_type=None):
        self._db_cur.execute(query, variables)
        if fetch_type:
            return getattr(self._db_cur, fetch_type)()
        else:
            self._db_connection.commit()

        
    def create_item_ehr_from_sql_queries(self, *args, **kwargs):
        if kwargs:
            q = kwargs['query']
            item = self.query(q, *args, fetch_type="fetchall")
        else:
            q = self.string_query
            item = self.query(q, *args, fetch_type="fetchall")
        
        try:
            _, ts, dicts = zip(*item)
            item_ehr = dict(zip(ts, dicts))
        except:
            item_ehr = {}

        return item_ehr
   
    def __del__(self):
        self._db_connection.close()
