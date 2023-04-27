import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from distutils.util import strtobool
from unplanned_net.dataset.datasource import TextDataSource
from unplanned_net.run.layers import EmbeddingLayers, AttentionPool, RecurrentLayer
from unplanned_net.utilities.parser import parse_nvidia_smi_processes 

class EhrAdmissionsModel(nn.Module):
    def __init__(self, datasources, args):
        super(EhrAdmissionsModel, self).__init__()
        self.args = args
        processes_running_on_gpu = parse_nvidia_smi_processes()
        
        self.device = 'cuda' if torch.cuda.is_available() and not processes_running_on_gpu and args.use_gpu else 'cpu'
        
        self.embeddings_net = EmbeddingLayers(datasources, args,)
        self.explainable_net = ExplainableNet(datasources, args,)

    def forward(self, inputs, seq_lens, baselines):
        embeddings = self.embeddings_net(inputs)
        out = self.explainable_net(*embeddings, seq_lens=seq_lens, baselines=baselines)    
        return out

class ExplainableNet(nn.Module):
    def __init__(self, feat_list, args,):
        super().__init__()
        
        self.args = args
        self.single_feat_models = torch.nn.ModuleList()
        self.feat_list = feat_list
        input_rnn = 0
        for feat_type in feat_list:
            net = SingleDomainNet(feat_type, args) 
            self.single_feat_models.add_module(net.name, net)
            input_rnn+=net.hidden_size
        
        self.class_pred = nn.Linear(input_rnn, args.num_events)

    def forward(self, *inputs, seq_lens=None, baselines=None):
        x = [m(i, seq_lens) for m,i in zip(self.single_feat_models, inputs)]
        x = torch.cat(x, axis=-1)
        o = self.class_pred(x)
        return o
        
class SingleDomainNet(nn.Module):
    def __init__(self, feat_type, args):
        super().__init__()
        self.args = args
        self.name = feat_type.name
        self.is_text_datasource = isinstance(feat_type, TextDataSource)
        if self.is_text_datasource:
            recurrent_args = {}
            for param in ['bidirectional', 'dropout', 'layers', 'recurrent_layer', 'units']:
                if feat_type.resumed_params:
                    recurrent_args[param] = feat_type.resumed_params[param]
                else:
                    recurrent_args[param] =  args.__dict__[param]
            self.embedding_size = feat_type.embedding_size
            
            if 'bidirectional' in recurrent_args and type(recurrent_args['bidirectional']) is str:
                recurrent_args['bidirectional'] = bool(strtobool(recurrent_args['bidirectional']))
            if recurrent_args['bidirectional'] and not (recurrent_args['units']%2==0):
                recurrent_args['units'] += 1
            self.hidden_size = recurrent_args['units']

            self.recurrent_layer = RecurrentLayer(self.name, input_size=self.embedding_size, 
                                                **recurrent_args)

            self.pool = AttentionPool(recurrent_args['units'])
            self.dropout = nn.Dropout(recurrent_args['dropout'])
        else:
            self.embedding_size =1
            self.hidden_size = 1
            self.pool = None
            self.dropout = None
    
    def forward(self, x, seq_lens):
        if self.is_text_datasource:
            x = self.recurrent_layer(x, seq_lens)
            x = self.pool(x)
            x = self.dropout(x)
        return x