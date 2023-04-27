import numpy as np
import torch.nn as nn 
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.special import logsumexp

### C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G


### C(t)-INDEX CALCULATION: this account for the weighted average for unbaised estimation
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0

    for i in range(N):
        try:

            tmp_idx = np.where(G[0,:] >= T_test[i])[0]

            if len(tmp_idx) == 0:
                W = (1./G[1, -1])**2
            else:
                W = (1./G[1, tmp_idx[0]])**2

            A[i, np.where(T_test[i] < T_test)] = 1. * W
            Q[i, np.where(Prediction[i] > Prediction)] = 1. # give weights

            if (T_test[i]<=Time and Y_test[i]==1):
                N_t[i,:] = 1.
        except:
            raise Exception
    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        print ("Warning: possible error in c-index")
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

def get_c_index(pred, event, time, num_events, num_categories, eval_time):

    ### EVALUATION
    va_result1 = np.zeros([num_events, len(eval_time)])
    event = event[:, np.newaxis]
    time = time[:, np.newaxis]

    for t, t_time in enumerate(eval_time):
        eval_horizon = int(t_time)

        if eval_horizon >= num_categories:
            print('ERROR: evaluation horizon is out of range')
            va_result1[:, t] = va_result2[:, t] = -1
        else:
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
            for k in range(num_events):
                va_result1[k, t] = c_index(risk[:,k], time, (event[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                #va_result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], time, (event[:,0] == k+1).astype(int), eval_horizon)
    tmp_valid = np.mean(va_result1)

    return tmp_valid

def get_lifelines_c_index(pred, event, time, num_events):
    event = event[:, np.newaxis]
    time = time[:, np.newaxis]
    
    c_index = [] 
    for k in range(num_events):
        k_pred = np.expand_dims(np.argmax(pred[:,k], axis=-1), axis=-1)
        k_c_index = concordance_index(time, k_pred, np.sign(event))
        c_index.append(k_c_index)
    return {'c_index':np.mean(c_index)}


def get_auc_dict(output, label, time_to_event, num_events, eval_time):
    d = dict()
    for l in range(num_events):
        for et in eval_time:
            target = (time_to_event<et).astype('int32')
            pred = np.sum(output[:,l,:et], axis=-1)
            try:
                auc = roc_auc_score(target, pred)
            except:
                auc = np.nan
            d.update({'auc_label-{}_t-{}'.format(l, et):auc})
    return d

def get_mcc_dict(output, label, time_to_event, num_events, eval_time):
    d = dict()
    for l in range(num_events):
        for et in eval_time:
            target = (time_to_event<et).astype('int32')
            pred = np.sum(output[:,l,:et], axis=-1)
            try:
                mcc = matthews_corrcoef(target, (pred>0.5).astype('int32'))
            except:
                mcc = np.nan
            d.update({'mcc_label-{}_t-{}'.format(l, et):mcc})
    return d

def softmax(a, axis=None):
    """
    Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
    :param a: ndarray/tensor
    :param axis: axis to sum over; default (None) sums over everything
    """
    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    return np.exp(a - lse)
