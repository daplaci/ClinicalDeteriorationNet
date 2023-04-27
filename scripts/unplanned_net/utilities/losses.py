import torch
import torch.nn as nn

_EPSILON = 1e-08

def l1(out, mask1_t, event):
    not_censored = torch.sign(event)
    l1 = torch.sum(torch.sum(torch.mul(mask1_t, out), dim=-1), dim=-1).unsqueeze(-1)
    l1 = torch.mean(-torch.log(l1 + _EPSILON))
    return l1

def rank_loss(out, mask2_t, event, time, device, num_events):
    sigma = 0.1
    eta = []
    
    for e_idx in range(num_events):
        like_time = torch.ones_like(time.unsqueeze(-1)).type(torch.FloatTensor).to(device)
        i2 = (event==(e_idx + 1)).type(torch.FloatTensor).to(device)
        i2 = torch.diag(i2.squeeze())
        event_out = out[:,e_idx,:]
        
        # R is the matrix of difference between the F at time s_i for x_i and x_j (i,j --> row,columns) vedi formula paper
        R = torch.matmul(event_out, mask2_t.permute(1,0))
        diag_R = R.diag().unsqueeze(0)
        R = torch.matmul(like_time, diag_R) - R
        R = R.permute(0,1)
        
        # T is the 
        T = torch.relu(torch.sign(torch.matmul(like_time, time.unsqueeze(-1).permute(1,0)) - \
            torch.matmul(time.unsqueeze(-1), like_time.permute(1,0))))
        T = torch.matmul(i2, T)

        e_eta = torch.mean(T * torch.exp(-R/sigma), dim=1) #keep i fixed and check the mean for all the j 
        eta.append(e_eta)
    
    eta = torch.stack(eta, axis=1)
    eta_mean = torch.mean(eta, dim=0) # mean in the batch
    rank_loss = torch.sum(eta_mean) #sum over the possible num Events
    return rank_loss

#calibration loss

#total loss
def total_loss(out, mask1_t, mask2_t, event, time, device, args):
    if args.time_to_event:
        return args.alpha_loss*l1(out, mask1_t, event) + args.beta_loss*rank_loss(out, mask2_t, event, time, device, args.num_events)
    elif args.binary_prediction:
        return nn.BCEWithLogitsLoss()(out.squeeze(), event)
    else:
        return nn.CrossEntropyLoss()(out, event.long()-1)