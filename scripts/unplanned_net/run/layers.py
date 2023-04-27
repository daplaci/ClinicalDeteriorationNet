import torch
import torch.nn as nn
from unplanned_net.dataset.notes_datasource import NotesDataSource
from unplanned_net.dataset.datasource import NumericalDataSource 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EmbeddingLayers(nn.Module):
    def __init__(self, feat_list, args, one_hot=False):
        super().__init__()
        self.args = args
        self.feat_list = feat_list
        self.single_embeddings = torch.nn.ModuleDict()
        for feat_type in feat_list:
            if isinstance(feat_type, NumericalDataSource):
                continue
            elif one_hot:
                embedding_layer = OneHotLayer(feat_type.vocab.n_words)
            elif isinstance(feat_type, NotesDataSource):
                embedding_layer = nn.Embedding.from_pretrained(
                    embeddings =  torch.FloatTensor(feat_type.em),
                    freeze=True,
                    padding_idx=0,
                    scale_grad_by_freq=True
                )
            else:
                embedding_layer = nn.Embedding(feat_type.vocab.n_words, 
                feat_type.embedding_size, padding_idx=0)
            
            self.single_embeddings[feat_type.name] = (embedding_layer)

    def forward(self, inputs):
        x = []
        for f in self.feat_list:
            if f.name in self.single_embeddings:
                x.append(self.single_embeddings[f.name](inputs[f.name]))
            else:
                i = inputs[f.name].float()
                x.append(i.unsqueeze(-1))
        x = tuple(x)
        return x
        

class OneHotLayer(nn.Module):
    def __init__(self, vocab_size, pool='max'):
        super(OneHotLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.functional.one_hot
        self.pool = pool

    def forward(self, x):
        embed = self.embed(x.long(), self.vocab_size)
        assert len(embed.size()) ==3
        
        if self.pool == 'max':
            embed, _ = torch.max(embed, dim=1)
        else:
            raise Exception ("This pool {} in OneHotLayer is not implemented yet.".format(self.pool))

        return embed.float()

class RecurrentLayer(nn.Module):
    def __init__(self, name, input_size=None, units=None, bidirectional=False, dropout=0.0, layers=1, recurrent_layer='lstm'):
        super().__init__()
        rnn = nn.LSTM if recurrent_layer =='lstm' else nn.GRU
        self.name = name
        self.hidden_size = units if not bidirectional else units//2

        self.rnn = rnn(input_size=int(input_size), hidden_size=self.hidden_size, batch_first=True, 
                        bidirectional=bidirectional, dropout=dropout, num_layers=layers)

    def pack_rnn_padd(self, x, seq_lens):
        x = pack_padded_sequence(x, lengths=seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.rnn(x)
        o, _ = pad_packed_sequence(o, batch_first=True)
        return o

    def forward(self, x, seq_lens):
        if seq_lens is not None:
            if len(x.size())>3:
                b, seq_len, pad_len, *_ = x.size()
                seq_lens = seq_lens[self.name].view(b*seq_len,1).squeeze()
                x = x.view(b*seq_len, pad_len, -1)
                o = self.pack_rnn_padd(x, seq_lens)
                o = o.view(b, seq_len, pad_len, -1)
            else:
                o = self.pack_rnn_padd(x, seq_lens[self.name])
        else:
            if len(x.size())>3:
                b, seq_len, pad_len, *_ = x.size()
                x = x.view(b*seq_len, pad_len, -1)
                o, *_ = self.rnn(x)
                o = o.view(b, seq_len, pad_len, -1)
            else:
                o, *_ = self.rnn(x)
        return o

class AttentionPool(nn.Module):
    def __init__ (self, hidden_size):
        super(AttentionPool, self).__init__()
        self.hidden_size = hidden_size
        self.attention_fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-2)
    
    def forward(self, x): #x size : B, SEQ_LEN, PADD_SEQ,  H
        attention_scores = torch.tanh(self.attention_fc(x)) #B, SEQ_LEN, PADD_SEQ, 1
        attention_scores = self.softmax(attention_scores) #B, SEQ_LEN, PADD_SEQ, 1
        x = torch.matmul(x.transpose(-2,-1), attention_scores) #B, SEQ_LEN, H, 1
        x = torch.sum(x, dim=-1)
        return x