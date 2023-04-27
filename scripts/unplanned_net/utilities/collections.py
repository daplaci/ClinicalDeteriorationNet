import optuna
from optuna.trial import TrialState
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional
from typing import Tuple
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def get_best_exp_optuna(studies, auc_table=None):
    """
        Plot the result of the grd search with optuna. save in figures/
    """
    storage = optuna.storages.RDBStorage(
                            url="postgresql://daplaci@trans-db-01/daplaci?options=-c%20search_path=unplanned_icu", 
                                                    engine_kwargs={"pool_size":0})
    best_exp_id = []
    for study_id in studies:
        study = optuna.load_study(study_name=study_id, storage=storage)
        exp_id = study.best_trial.user_attrs['exp_id']
        print (f"Best params for study {study_id} from optuna storage are {study.best_params} with exp_id {exp_id}")
        best_exp_id.append(exp_id)
    return best_exp_id

def get_best_param_optuna(study_id, auc_table=None):
    """
        get best param for study
    """
    storage = optuna.storages.RDBStorage(
                            url="postgresql://daplaci@trans-db-01/daplaci?options=-c%20search_path=unplanned_icu", 
                                                    engine_kwargs={"pool_size":0})

    study = optuna.load_study(study_name=study_id, storage=storage)
    params = deepcopy(study.best_params)
    params.update(study.best_trial.user_attrs)
    return params

def get_best_from_auc_table(studies, metric='val_mcc', auc_table=None):
    assert 'val' in metric
    best_exp_id = []
    if auc_table is None:
        auc_table = pd.read_csv('AUC_history_gridsearch.tsv', sep='\t')
    for study_id in studies:
        study_table = auc_table[auc_table.study_id == study_id]
        best_trial = study_table.loc[study_table[metric]== np.max(study_table[metric].values), 'exp_id']
        exp_id = best_trial.iloc[0]
        print (f"Best params for study {study_id} from auc table are {study_table[study_table.exp_id == exp_id].to_dict('records')}")
        best_exp_id.append(exp_id)
    return best_exp_id

def get_successfull_trials(study_id):
    assert type(study_id) is str

    storage = optuna.storages.RDBStorage(
                            url="postgresql://daplaci@trans-db-01/daplaci?options=-c%20search_path=unplanned_icu", 
                                                    engine_kwargs={"pool_size":0})

    study = optuna.load_study(study_name=study_id, storage=storage)
    trials = study.get_trials(states=(TrialState.COMPLETE,))
    trials = [t for t in trials if t.values[0] < np.inf and t.values[0]>0]
    return trials

def get_params_from_successfull_trial(study_id, exp_id):
    trials = get_successfull_trials(study_id)
    trial = [t for t in trials if t.user_attrs['exp_id'] == exp_id][0]
    return trial.params
    
class MaxTrialsCallback:
    """Set a maximum number of trials before ending the study.

    While the :obj:`n_trials` argument of :obj:`optuna.optimize` sets the number of trials that
    will be run, you may want to continue running until you have a certain number of successfullly
    completed trials or stop the study when you have a certain number of trials that fail.
    This :obj:`MaxTrialsCallback` class allows you to set a maximum number of trials for a
    particular :class:`~optuna.trial.TrialState` before stopping the study.

    Example:

        .. testcode::

            import optuna
            from optuna.study import MaxTrialsCallback
            from optuna.trial import TrialState


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                return x ** 2


            study = optuna.create_study()
            study.optimize(
                objective,
                callbacks=[MaxTrialsCallback(10, states=(TrialState.COMPLETE,))],
            )

    Args:
        n_trials:
            The max number of trials. Must be set to an integer.
        states:
            Tuple of the :class:`~optuna.trial.TrialState` to be counted
            towards the max trials limit. Default value is :obj:`(TrialState.COMPLETE,)`.
    """

    def __init__(
        self, n_trials: int, states: Tuple[TrialState, ...] = (TrialState.COMPLETE,)
    ) -> None:
        self._n_trials = n_trials
        self._states = states

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        trials = study.get_trials(deepcopy=False, states=self._states)
        n_complete = len(trials)
        if n_complete >= self._n_trials:
            study.stop()