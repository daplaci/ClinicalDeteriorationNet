import numpy as np
import pandas as pd
import random

### MASK FUNCTIONS
'''
    fc_mask2      : To calculate LOSS_1 (log-likelihood loss)
    fc_mask3      : To calculate LOSS_2 (ranking loss)
'''
def f_get_fc_mask2(time, label, num_events, num_categories):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_events, num_categories]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros((num_events, num_categories)) # for the first loss function
    if label != 0:  #not censored
        mask[int(label-1),int(time)] = 1
    else: #label[i,2]==0: censored
        mask[:,int(time+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask.astype(np.float32)


def f_get_fc_mask3(time, meas_time, num_categories):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_categories].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([num_categories]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        t = int(time) # censoring/event time
        mask[:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask.astype(np.float32)