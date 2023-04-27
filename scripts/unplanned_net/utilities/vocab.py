import pickle
import nltk
import re
PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
import sys
sys.path.append(PATH_TO_LIB)
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from functools import partial
from multiprocessing import Pool
from collections import Counter
import itertools
import psycopg2
import multiprocessing
from multiprocessing import Pool
import argparse
import itertools
import time
import os
from functools import partial 
import unplanned_net.utilities.parser as mainparser


wordtokenizer = RegexpTokenizer(r'\S+').tokenize

def get_stopwords(lang = 'danish', additional = ['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}']):
    # Define 'stop_words'
    stop_words = set(stopwords.words(lang))
    stop_words.update(additional)
    return stop_words

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def chunker(it, size):
    "Generates chunks of 'size' elements from iterator"
    it = iter(it)
    while True:
        tasks = [x for x in itertools.islice(it, size)]
        if not tasks:
            return
        yield tasks

class DefaultDict(dict):
    def __init__(self,name, default_factory, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.default_factory = default_factory

    def __setitem__(self, key, value):
        if key in self and self.name!='word2count':
            raise ("Error: Use add word method to fill the vocab")
        super(DefaultDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_factory()

def default_zero():
    return 0
def default_unk():
    return 'unk'

class Vocab:
    def __init__(self, name):
        print ("Generating vocab {}.".format(name))
        self.name = name
        self.word2index = DefaultDict('word2index', default_zero)
        self.word2count = DefaultDict('word2count', default_zero)
        self.index2word = DefaultDict('index2word', default_unk)
        self.add_word('unk')
        self.add_word('pad')

    def add_word(self, word, count=1):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = count
        else:
            self.word2count[word] += count
    
    @property
    def n_words(self):
        return self.index2word.__len__()

    def update_vocab(self, other_vocab, min_freq):
        for w in other_vocab.word2index:
            c = other_vocab.word2count[w]
            if c > min_freq:
                self.add_word(w, c)
        print (f"Vocab {self.name} with new min freq \
        {min_freq} has {self.n_words} words.")

class VocabExtractorWorker(object):

    def __init__(self, name, args=None):

        self.con = psycopg2.connect("host=trans-db-01 dbname=daplaci user=daplaci")

        self.vocab_name = name
        if self.vocab_name == "notes":
            self.tf_table = "notestable"
        else:
            self.tf_table = "jsontable"
        self.wc = Counter()
        if self.vocab_name == "diag":
            self.processing_function = partial(get_disease_code, level_code=args.level_code)
        elif self.vocab_name == "biochem":
            biochem_values = pickle.load(open(os.path.join(args.input_dir, "biochem_values.pkl"), 'rb'))
            biochem_top = pickle.load(open(os.path.join(args.input_dir, "biochem_top.pkl"), 'rb'))
            self.processing_function = partial(get_npu_code, 
                                                top_biochem=args.top_biochem, 
                                                biochem_bins=args.biochem_bins, 
                                                include_percentile= args.include_percentile, 
                                                biochem_values=biochem_values, biochem_top=biochem_top)
        elif self.vocab_name == "notes":
            self.processing_function = lambda x:x

    def extract(self, task):
        raw_words, = task
        if raw_words:
            words = [self.processing_function(w) for w in raw_words]
        else:
            words = []
        self.wc.update(Counter(words)) 

    def __call__(self, tasks):

        fetch_words = f"""SELECT data->'{self.vocab_name}'
                    FROM {self.tf_table} WHERE pid = ANY(%s);"""

        with self.con.cursor() as cur1:

            cur1.execute(fetch_words, (tasks, ))  # send query

            # extract from fetched entries
            [self.extract(task) for task in cur1]

        return None

def get_disease_code(code, level_code=None):
    if level_code:
        if code[0].isdigit():
            return code[:level_code]
        else:
            return code[:level_code+1]
    else:
        return code

def get_npu_code(npu_code, top_biochem=250, biochem_bins=10, include_percentile= True, biochem_values=None, biochem_top=None):
    # here you process the npu code from the biochem table
    biochem, value = npu_code.split("@")

    if biochem_top[biochem] > top_biochem:
        return "unk"
    else:
        if include_percentile:
            if is_number(value):
                numerical_index = sum(float(value) > biochem_values[biochem][biochem_bins])
                lower_bound = max(numerical_index-1, 0)
                upper_bound = min(lower_bound + 1, len(biochem_values[biochem][biochem_bins])-1)
                code = "{}@{}-{}".format(biochem , biochem_values[biochem][biochem_bins][lower_bound], biochem_values[biochem][biochem_bins][upper_bound])
            else:
                code = "{}@{}".format(biochem , re.sub(r'[^\w\s]','',value))
        else:
            code = biochem
    return code

def child_initialize(_worker, _data_source, _args):
    global VocabExtractorWorker, data_source, args
    VocabExtractorWorker = _worker
    args = _args
    data_source = _data_source

def subprocess(pid_tasks):
    worker = VocabExtractorWorker(data_source, args)
    for chunk in chunker(pid_tasks, 100):
        worker(chunk)

    return worker.wc

def generate_vocab_from_table(data_source, num_workers, args=None):
    DSN = "host=trans-db-01 dbname=daplaci user=daplaci"
    if data_source == "notes":
        table = "notestable"
    else:
        table = "jsontable"
    con = psycopg2.connect(DSN)

    # main document frequency counter
    vocab = Vocab(data_source)
    print ("Extracting vocab from database")
    start = time.time()
    with con.cursor("cur1") as cur1:

        cur1.itersize = 10000  # only with named cursor
        cur1.execute("SELECT DISTINCT pid FROM {}".format(table))
        with multiprocessing.Pool(max(1, num_workers), 
                                initializer=child_initialize, 
                                initargs=(VocabExtractorWorker, data_source, args)) as pool:
            pid_tasks = chunker(cur1, size=10000)  # 1 million
            for wc in pool.imap_unordered(subprocess, pid_tasks):
                for term, count in wc.items():
                    vocab.add_word(term, count)

    print ("Time elapsed -- {}".format(time.time() - start))
    return vocab
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_source', type=str, default="diag", help='data file')
    parser.add_argument('--num_cores', type=int, default=100, help='data file')
    args = mainparser.parse_args(parent_parser=[parser])
    
    vocab = generate_vocab_from_table(args.data_source, args.num_cores, args)
    
    with open('input/{}_vocab.pkl'.format(vocab.name), 'wb') as f:
        print ("Saving vocab {}..".format(vocab.name))
        pickle.dump(vocab, f)
