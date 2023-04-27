from argparse import ArgumentParser
import os
from typing import List, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from unplanned_net.utilities.database import MyDB
from unplanned_net.utilities.vocab import Vocab
from unplanned_net.utilities.collections import get_best_param_optuna

DATASOURCES = {}

def padder(x, max_len, tkn='pad'):
    if type(x) is list and len(x)>0 and type(x[0]) is list:
        x = [padder(el, max_len, tkn) for el in x]
        return x
    else:
        if hasattr(x, '__iter__'):
            x = list(x)
        else:
            x = [x]
    return [tkn]*(max_len - len(x)) + x[-max_len:]

def compute_embedding(vocab_size, embedding_coeff):
    emb_dim = int(round(np.sqrt(np.sqrt(vocab_size))*6*embedding_coeff))
    if emb_dim%2!=0:
        emb_dim +=1
    return emb_dim

def register_datasource(ds_name):
    def decorator(f):
        DATASOURCES[ds_name] = f
        return f
    return decorator

class NumericalDataSource(ABC):

    def set_stats(self, stats):
        print (f"Setting stats for {self.name}")
        self.stats = stats[self.name] if self.name in stats else None

    def get2d_input(self, value, *args):
        x = np.float32(value - self.stats["mean"])/self.stats["std"]
        return x

class TextDataSource(ABC):

    @abstractmethod
    def __init__(self, args: ArgumentParser, extract_full_text: bool) -> None:
        
        self.args = args
        self.ds_argument = str(self.args.__dict__[self.name])
        self.text_to_vocabularies = np.vectorize(self.get_tkn)
        
        print (f"Data source {self.name} running !")
        
        self.resumed_params = self.resume_datasource_arguments()
        std_vocab = self.get_vocab()
        self.vocab = Vocab(self.name)
        if isinstance(self.resumed_params, dict) and \
                "min_freq" in self.resumed_params:
            self.vocab.update_vocab(std_vocab, self.resumed_params["min_freq"])
        else:
            self.vocab.update_vocab(std_vocab, args.min_freq)
        self.padd_size = self.get_padd_size(extract_full_text)

        if self.resumed_params:
            embedding_coeff = self.resumed_params['embedding_coeff']
        else:
            embedding_coeff = self.args.embedding_coeff
        
        self.embedding_size = compute_embedding(
            self.vocab.n_words, 
            embedding_coeff
            )


    @abstractmethod
    def get_vocab(self) -> Vocab:
        pass
    
    @abstractmethod
    def apply_word_filter(self):
        pass
    
    def get_data_for_stats(self, database_obj : MyDB, pid: str, time_limit: float, time_adm: float):
        data = database_obj.create_item_ehr_from_sql_queries(pid, time_limit, time_adm, query=self.admission_query)
        x_text = []
        for t in sorted(data):
            if data[t]:
                x_text.extend(data[t])
        return x_text

    @abstractmethod #this may no longer be defined as abstract as it is not overriden in any classes
    def get2d_input(self, database_obj : MyDB, pid: str, time_limit: float):
        data = database_obj.create_item_ehr_from_sql_queries(pid, time_limit, query=self.query)
        x_text = []
        for t in sorted(data):
            if data[t]:
                x_text.extend(data[t])
        if self.padd_size:
            padd_size = self.padd_size
        else:
            #this condidition applys when no padding is used (only for data extraction)
            padd_size = max(1, len(x_text))
        
        seq_lens = max(min(padd_size, len(x_text)),1)
        x_text = padder(x_text, padd_size)
        x = self.text_to_vocabularies(np.array(x_text))
        return x_text, x, seq_lens
    
    def get_tkn(self, x: str) -> int:
        x  = self.apply_word_filter(x)
        
        if self.resumed_params and 'min_freq' in self.resumed_params:
            min_freq = self.resumed_params['min_freq']
        else:
            min_freq = self.args.min_freq
        
        if self.vocab.word2count[x] > min_freq:
            idx = self.vocab.word2index[x]
        else:
            idx = self.vocab.word2index["unk"]
        return idx

    def resume_datasource_arguments(self, ) -> Union[dict, None]:

        resumed_grid_path = os.path.join(
                self.args.base_dir, 
                self.ds_argument
                )
                
        if not os.path.exists(resumed_grid_path):
            print (f"Datasource {self.name} log: \
                no experiment found at {resumed_grid_path}")
            return None

        print (f"Loading datasource {self.name} \
            from pre-existing folder {self.ds_argument}")
        results_table = os.path.join(resumed_grid_path, 'AUC_history_gridsearch.tsv')
        auctable = pd.read_csv(results_table, sep='\t')
        studies = auctable.study_id.tolist()
        assert len(set(studies)) == 1
        best_params = get_best_param_optuna(studies[0])
        return best_params

    def get_padd_size(self, extract_full_text: bool) -> Union[int, None]:
        padd_name = f"padd_{self.name}"
        if self.resumed_params:
            return self.resumed_params[padd_name]
        elif (padd_name in self.args.__dict__) and (not extract_full_text):
            return self.args.__dict__[padd_name] 
        else:
            return None

def get_datasources(args: ArgumentParser, ) -> \
    List[Union[NumericalDataSource, TextDataSource]]:
    
    selected_datasources = []
    print (sorted(DATASOURCES))
    for ds_name in sorted(DATASOURCES):
        if args.__dict__[ds_name]:
            datasource_class = DATASOURCES[ds_name]
            selected_datasources.append(datasource_class(args))
    return selected_datasources