import numpy as np
from functools import partial
import multiprocessing
import json
import sys,os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import unplanned_net.dataset.data_utils as data_utils
import unplanned_net.utilities.parser as parser
from unplanned_net.utilities.database import MyDB
from unplanned_net.utilities.vocab import *
from unplanned_net.dataset.datasource import get_datasources, NumericalDataSource

def subprocess(tasks, data=None):
    with open(os.path.join(args.input_dir, f"{subdataset.split_group}_{data}.txt"), 'a+') as f:
        for t in tasks:
            _, inputs,  _, _, _, label,  _ = subdataset.__getitem__(t)
            x_text = ' '.join(inputs[data])
            row = "__label__{} {}\n".format(label, x_text)
            f.write(row)
    return None

def child_initialize(_dataset, _args, dbcon):
    global subdataset, args
    subdataset = _dataset
    subdataset.con = dbcon()
    args = _args

def save_source(data):
    print ("saving source {}".format(data))
    with open(os.path.join(args.input_dir, f"{train_dataset.split_group}_{data}.txt"), 'w') as f:
        pass
    
    range_train = range(train_dataset.__len__())
    tasks = np.array_split(range_train, 100)
    psubprocess = partial(subprocess, data=data)
    #
    with multiprocessing.Pool(100, initializer=child_initialize, 
        initargs=(train_dataset,  args, MyDB)) as p:
        p.map(psubprocess, tasks)

    print (f"Done with train data {data}..")

    with open(os.path.join(args.input_dir, f"{val_dataset.split_group}_{data}.txt"), 'w') as f:
        pass

    range_val = range(val_dataset.__len__())
    tasks = np.array_split(range_val, 100)
    psubprocess = partial(subprocess, data=data)

    with multiprocessing.Pool(100, initializer=child_initialize, 
        initargs=(val_dataset, args, MyDB)) as p:
        p.map(psubprocess, tasks)

    print (f"Done with val data {data}..")

    with open(os.path.join(args.input_dir, f"{test_dataset.split_group}_{data}.txt"), 'w') as f:
        pass

    range_test = range(test_dataset.__len__())
    tasks = np.array_split(range_test, 100)
    psubprocess = partial(subprocess, data=data)
    with multiprocessing.Pool(100, initializer=child_initialize, 
        initargs=(test_dataset, args, MyDB)) as p:
        p.map(psubprocess, tasks)

    print (f"Done with test data {data}..")

if __name__ == '__main__':

    args = parser.parse_args()

    datasources = get_datasources(args)
    adm_data, ehr_data, t_person = data_utils.get_input_data(args)

    np.random.seed(args.seed)
    for p in adm_data:
        adm_data[p]['split_group'] = np.random.choice(['train','val'], p=[.80,.20])

    adm_test = json.load(open(os.path.join(args.input_dir, args.admissions_test),'r'))
    for p in sorted(adm_test):
        adm_test[p]['split_group'] = "test"

    train_dataset, train_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 
                                'train', args, datasources=datasources, use_weights_sampler=True)

    val_dataset, val_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 
                            'val', args, datasources=datasources, use_weights_sampler=False, stats=train_dataset.stats)

    test_dataset, test_loader = data_utils.get_dataclasses(adm_test, ehr_data, t_person, 
                            'test', args, datasources=datasources, use_weights_sampler=False, stats=train_dataset.stats)

    for ds in datasources:
        if isinstance(ds, NumericalDataSource):
            ds.set_stats(train_dataset.stats)

    print ("Saving fasttext input.. \n")

    if args.notes:
        save_source("notes")
    if args.biochem:
        save_source("biochem")
    if args.diag:
        save_source("diag")