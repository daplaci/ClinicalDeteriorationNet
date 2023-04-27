import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import DanishStemmer
from nltk.corpus import stopwords
import psycopg2
import itertools
import multiprocessing
import io
import sys
from tqdm import tqdm
from collections import Counter
import datetime
import json
from nltk.tokenize import sent_tokenize, word_tokenize

CONNECT_STRING = "" #define here as user database and port (pw if not in .pgpass)

def chunker(it, size):
    "Generates chunks of 'size' elements from iterator"
    it = iter(it)
    while True:
        tasks = [x for x in itertools.islice(it, size)]
        if not tasks:
            return
        yield tasks


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


class Worker(object):

    def __init__(self):
        
        self.bth_con = psycopg2.connect("host=trans-db-01 user=daplaci dbname=bth")
        self.con = psycopg2.connect("host=trans-db-01 user=daplaci dbname=daplaci")
        self.dest_table = 'notestable'
        
        self.negation_window_range = range(6)#range(snakemake.params['negation_window'] + 1)
        self.min_token_length = 4 #snakemake.params['min_token_length']
        self.regex_punctuation = "(?<!\d)[^\w\s](?!\d)|(?<=\d)[^\w\s](?!\d)|(?<!\d)[^\w\s](?=\d)|(?<=\d)[^\w\s.,](?=\d)"
        self.negations = {'ikke', 'ingen', "ej", "heller", "hverken", "kendt", "tidligere", "obs"}

    def extract(self, task):

        pid, rekvdt, txt_id, raw_text = task  # unpack
        text = self.remove_tail(raw_text)
        retained, negated = self.remove_negated_words__punctuation__whitespace__lower(text)
        retained = self.remove_names(retained)
        retained = self.remove_stopwords(retained)

        datetime, unix_timestamp = self.parse_date(rekvdt)
        row = "\t".join((str(pid), datetime, unix_timestamp,
                        json.dumps({"notes":retained}, ensure_ascii=False)))
        yield row

    def parse_date(self, date_str):
        
        if date_str == 'nan' or date_str == 'NA':
            raise Exception
        if len(date_str) == 10:
            format_str = '%Y-%m-%d'
        elif len(date_str) == 4:
            format_str = '%Y'
        elif len(date_str) == 12:
            format_str = '%Y%m%d%H%M'
        elif len(date_str) == 20:
            format_str = "%Y-%m-%dT%H:%M:%SZ"
        else:
            format_str = '%Y-%m-%d %H:%M'
            
        dt = datetime.datetime.strptime(date_str, format_str)
        timestamp = datetime.datetime.timestamp(dt)
        return str(dt), str(timestamp)

    def clean_csv_value(self, value):
        
        if value is None:
            return r'\N'
        value = str(value).replace(';', ',')
        value = value.replace('\\', '')
        return value

    def remove_tail(self, text):
        
        regex_tail = "\.[^.]{2,50}/.{3,30}$"
        text_no_tail = re.sub(regex_tail, "", text)
        return text_no_tail

    def remove_negated_words__punctuation__whitespace__lower(self, text):
        
        text = re.sub("\s+", " ", text)

        retained_tokens = []
        negated_tokens = []
        
        for sent in sent_tokenize(text, language = "danish"):
            words = word_tokenize(sent, language = "danish")
            words = [re.sub(self.regex_punctuation, "", w).lower() for w in words]
            #words = [w for w in words if w not in ("", " ")]

            negated_idx = [i+j for i,x in enumerate(words) for j in self.negation_window_range if x in self.negations]
            retained_tokens.extend(w for k,w in enumerate(words) if k not in negated_idx and len(w) >= self.min_token_length)
            negated_tokens.extend(w for k,w in enumerate(words) if k in negated_idx and k < len(words))

        return retained_tokens, negated_tokens
        
    def remove_stopwords(self, tokens):

        tokens_wo_stopwords = [t for t in tokens if t not in danish_stop_words]
        return tokens_wo_stopwords

    def remove_names(self, tokens):
        
        tokens_wo_names = [t for t in tokens if t not in names]
        return tokens_wo_names

    def __call__(self, tasks):

        # query for getting notes 
        fetch_data = """
            SELECT pid, rekvdt, txt_id, text_t1 FROM epj_dev.txt
                WHERE txt_id=ANY(%s) and text_t1 <> ''
        """

        with self.bth_con.cursor() as cur1:

            cur1.execute(fetch_data, (tasks, ))  # send query

            # extract from fetched entries
            results = (row for task in cur1 for row in self.extract(task))
            output = PseudoFile(results)

            with self.con.cursor() as cur2:
                cur2.copy_from(output, self.dest_table, sep="\t")

        self.con.commit()

        return None


def subprocess(tasks):
    worker = Worker()
    for chunk in chunker(tasks, 1000):
        worker(chunk)

    return None

def child_initialize(_worker, _names, _danish_stop_words):
    global Worker, names, danish_stop_words
    Worker = _worker
    names = _names
    danish_stop_words = _danish_stop_words


if __name__ == "__main__":
    print("###\tLoading stopwords")
    danish_stop_words = set(stopwords.words('danish'))

    print("###\tLoading names")
    names = pd.read_csv(
        "out_all_names_combined_updated.tsv", 
        usecols=[0], squeeze=True, sep='\t', error_bad_lines=False, header=None
    )
    names = {x.lower() for x in names.map(str)}
    
    print ("###\tTable Init.. \n")
    con = psycopg2.connect(CONNECT_STRING)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS notestable;")
    con.commit()

    create_table_query = """CREATE TABLE notestable (pid text,datetime timestamp,ts double precision,data jsonb);"""
    cur.execute(create_table_query)
    con.commit()
    con.close()

    print ("###\tFill table.. \n")
    DSN = "host=trans-db-01 user=daplaci dbname=bth"
    con = psycopg2.connect(DSN)

    with con.cursor("cur1") as cur1:

        cur1.itersize = 10000  # only with named cursor
        cur1.execute("SELECT txt_id FROM epj_dev.txt")

        with multiprocessing.Pool(100, 
            initializer=child_initialize, 
            initargs=(Worker, names, danish_stop_words)) as pool:
            tasks = chunker(cur1, size=1000000)  # 1 million
            pool.map(subprocess, tasks)
    
    con.commit()
    con.close()
    
    DSN = "host=trans-db-01 user=daplaci dbname=daplaci"
    con = psycopg2.connect(DSN)
    
    cur = con.cursor()
    cur.execute("CREATE INDEX notestable_pid_index ON notestable (pid);") 

    con.commit()
    con.close()