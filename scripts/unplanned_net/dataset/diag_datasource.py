import numpy as np
import pickle
import os
from unplanned_net.utilities.database import MyDB
from unplanned_net.dataset.datasource import TextDataSource, compute_embedding, padder, register_datasource
from unplanned_net.utilities.vocab import generate_vocab_from_table, get_disease_code, Vocab
from unplanned_net.utilities.parser import get_argument_parser
from argparse import ArgumentParser

@register_datasource("diag")
class DiagnosisDataSource(TextDataSource):
    name = "diag"
    
    def __init__(self, args: ArgumentParser, extract_full_text: bool = None):        
        super().__init__(args, extract_full_text)
        self.query = f"""
SELECT pid, ts, data -> '{self.name}' as {self.name}
from jsontable where pid=%s and ts <= %s """
        # this line below is not actually doing anything
        # just avoids to raise error for missing params in the query from __get_ds_from_admissions_to_timelimit__
        self.admission_query = self.query + " and ts <= %s" 

    def apply_word_filter(self, word: str):
        if word!='pad':
            if self.resumed_params:
                word = get_disease_code(word, level_code=self.resumed_params['level_code'])
            else:
                word = get_disease_code(word, level_code=self.args.level_code)
        return word

    def get_vocab(self) -> Vocab:
        parser = get_argument_parser() 
        generic_ds_vocab = os.path.join(self.args.input_dir, 'diag_vocab.pkl')

        if self.resumed_params:
            try:
                vocab_path = os.path.join(self.args.base_dir, 
                    self.ds_argument, 
                    f"best_weights/{self.resumed_params['exp_id']}.vocab.{self.name}.pkl")
                vocab = pickle.load(open(vocab_path, 'rb'))
                print (f"Loaded vocab from {vocab_path}")
            except FileNotFoundError:
                assert self.resumed_params['level_code'] == parser.get_default("level_code")
                print (f"""WARNING: vocab not found at {vocab_path}.
                This might mean all the parameters are default of there is a problem
                in the vocab exp id.""")
                vocab = pickle.load(open(generic_ds_vocab, 'rb'))
            return vocab
        
        if parser.get_default("level_code") != self.args.level_code:
            custom_ds_vocab = f"best_weights/{self.args.exp_id}.vocab.{self.name}.pkl"
            if os.path.exists(custom_ds_vocab):
                vocab = pickle.load(open(custom_ds_vocab, 'rb'))
            else:
                vocab = generate_vocab_from_table("diag", self.args.num_workers*2, self.args)
                with open(custom_ds_vocab, 'wb') as f:
                    pickle.dump(vocab, f)
        else:
            vocab = pickle.load(open(generic_ds_vocab, 'rb'))
        print ("Loaded Diag vocab. Number of words :{}\n".format(vocab.n_words))
        return vocab

    def get2d_input(self, database_obj : MyDB, pid: str, time_limit: float):
        return super().get2d_input(database_obj, pid, time_limit)