import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_model(datasources, args):
    def count_parameters(model, requires_grad=True):
        return sum(p.numel() for p in model.parameters() if p.requires_grad is requires_grad)
    
    if args.model == 'deep':
        from unplanned_net.run.deep import EhrAdmissionsModel
        model = EhrAdmissionsModel(datasources, args)
    elif args.model in ['lr', 'mlp']:
        from unplanned_net.run.mlp import MLPAdmissionsModel
        model = MLPAdmissionsModel(datasources, args)
    else:
        raise Exception("Model name not recognized")

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print('The model has {} trainable parameters'.format(count_parameters(model)))
    print('The model has {} non-trainable parameters'.format(count_parameters(model, requires_grad=False)))
    print ("Loading Model {} on device: {}".format(args.model, model.device))
    
    model = model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_regularizer)
    return model, optimizer
