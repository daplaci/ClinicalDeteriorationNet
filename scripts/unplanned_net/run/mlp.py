from typing import List
import torch
import torch.nn as nn
from unplanned_net.dataset.datasource import TextDataSource
from unplanned_net.run.layers import AttentionPool, EmbeddingLayers
from unplanned_net.utilities.time import timeit
from unplanned_net.utilities.parser import parse_nvidia_smi_processes 

class MLPAdmissionsModel(nn.Module):
    def __init__(self, datasources, args):
        super(MLPAdmissionsModel, self).__init__()
        self.args = args
        processes_running_on_gpu = parse_nvidia_smi_processes()
        
        self.device = 'cuda' if torch.cuda.is_available() and not processes_running_on_gpu and args.use_gpu else 'cpu'

        self.embeddings_net = EmbeddingLayers(datasources, args, one_hot=True)
        self.explainable_net = ExplainableNet(datasources, args,)

    def forward(self, inputs, *_):
        embeddings = self.embeddings_net(inputs)
        out = self.explainable_net(*embeddings)    
        return out

class ExplainableNet(nn.Module):
    def __init__(self, feat_list: List, args,):
        super().__init__()
        
        self.args = args
        self.single_feat_models = torch.nn.ModuleList()
        self.feat_list = feat_list
        input_size = 0
        for feat_type in feat_list:
            if isinstance(feat_type, TextDataSource):
                input_size += feat_type.vocab.n_words
            else:
                input_size += 1

        self.class_pred = nn.Linear(input_size, args.num_events)

    def forward(self, *inputs): #TODO add pool option for MLP - num_layers 

        x = torch.cat(inputs, axis=-1)
        o = self.class_pred(x)
        return o