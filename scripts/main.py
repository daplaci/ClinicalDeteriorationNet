from os.path import dirname, realpath
import sys
PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
sys.path.append(PATH_TO_LIB)
import numpy as np
import unplanned_net.dataset.data_utils as data_utils
from unplanned_net.utilities.parser import parse_args
import unplanned_net.run.model_utils as model_utils
from unplanned_net.dataset.datasource import get_datasources, NumericalDataSource
import unplanned_net.run.learn as learn
from unplanned_net.utilities.writer import DF_writer
from unplanned_net.utilities.vocab import *

args = parse_args()
df_writer = DF_writer()

datasources = get_datasources(args)
adm_data, ehr_data, t_person = data_utils.get_input_data(args)

np.random.seed(args.seed)
for cv_num in range(args.cv_folds):
    
    args.cv_num = cv_num+1 
    
    for p in sorted(adm_data):
        adm_data[p]['split_group'] = np.random.choice(['train','val'], p=[.80,.20])

    args.weights_path = 'best_weights/{}_{}.pt'.format(args.exp_id, args.cv_num)

    train_dataset, train_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 
                                'train', args, datasources=datasources, use_weights_sampler=True)

    val_dataset, val_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 
                            'val', args, datasources=datasources, use_weights_sampler=True, stats=train_dataset.stats)
    
    for ds in datasources:
        if isinstance(ds, NumericalDataSource):
            ds.set_stats(train_dataset.stats)

    model, optimizer = model_utils.get_model(train_dataset.datasources, args)

    if not args.skip_train:
        train_hist = learn.run_epochs(train_loader, val_loader, model, optimizer, args)
    else:
        train_hist = None
    
    del val_dataset, val_loader
    val_dataset, val_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 
                            'val', args, datasources=datasources, use_weights_sampler=False, stats=train_dataset.stats)
    if not args.skip_val:
        df_writer.update_train_results(train_loader, val_loader, model, optimizer, train_hist, args)

