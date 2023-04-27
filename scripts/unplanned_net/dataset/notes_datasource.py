from argparse import ArgumentParser
from typing import Optional
import numpy as np
import pickle
import os
import fasttext
import sys
from unplanned_net.utilities.database import MyDB
from unplanned_net.dataset.datasource import TextDataSource, compute_embedding, register_datasource

from unplanned_net.utilities.vocab import Vocab

@register_datasource("notes")
class NotesDataSource(TextDataSource):
    name = "notes"

    def __init__(self, args: ArgumentParser, extract_full_text: bool = None):        
        super().__init__(args, extract_full_text)
        self.query = f"""
SELECT pid, ts, data -> '{self.name}' as {self.name}
from notestable where pid=%s and ts <= %s"""
        
        self.admission_query = self.query + " and ts>%s"
        self.em = self.build_embedding()
    
    def apply_word_filter(self, x):
        return x

    def get_vocab(self) -> Vocab:
        generic_ds_vocab = os.path.join(self.args.input_dir, 'notes_vocab.pkl')
        if self.resumed_params:
            vocab_path = os.path.join(self.args.base_dir, 
                self.ds_argument, 
                f"best_weights/{self.resumed_params['exp_id']}.vocab.{self.name}.pkl")
            try:
                vocab = pickle.load(open(vocab_path, 'rb'))
                print (f"Loaded vocab from {vocab_path}")
            except FileNotFoundError:
                print (f"""WARNING: no specific vocab exp if found for {vocab_path}. 
                Loading default at {generic_ds_vocab}""")
                vocab = pickle.load(open(generic_ds_vocab, 'rb'))
            return vocab
        
        vocab = pickle.load(open(generic_ds_vocab, 'rb'))
        print ("Loaded Notes vocab. Number of words :{}\n".format(vocab.n_words))
        return vocab
    
    def build_embedding(self) -> Optional[np.ndarray]:
        print (f"Loading embedding for {self.name}")
        pretrained_model = os.path.join(
            self.args.input_dir, 
            f"model_unsupervised_{self.name}.bin"
            )
        
        try:
            model = fasttext.load_model(pretrained_model)
        except (FileNotFoundError, ValueError) as e:
            print (f"No pretrained model found for {self.name}.\{e}")
            return None

        self.embedding_size = model.get_dimension()
        words = self.vocab.word2index.keys()    
        m = {i:np.zeros(self.embedding_size) for i in range(self.vocab.n_words)}   
        for w in words:
            v = model.get_word_vector(w)
            m[self.vocab.word2index[w]] = v
        em = np.asarray([m[k] for k in sorted(m)])
        return em
    

    def get2d_input(self, database_obj : MyDB, pid: str, time_limit: float):
        x_text, x, seq_lens = super().get2d_input(database_obj, pid, time_limit)
        return x_text, x, seq_lens
