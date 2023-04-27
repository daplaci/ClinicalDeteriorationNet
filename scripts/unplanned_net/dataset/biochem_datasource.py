from argparse import ArgumentParser
import pickle
import os
from unplanned_net.utilities.database import MyDB
from unplanned_net.dataset.datasource import TextDataSource, register_datasource
from unplanned_net.utilities.vocab import generate_vocab_from_table, get_npu_code, Vocab
from unplanned_net.utilities.parser import get_argument_parser

@register_datasource("biochem")
class BiochemDataSource(TextDataSource):
    name = "biochem"

    def __init__(self,  args: ArgumentParser, extract_full_text: bool = None):
        super().__init__(args, extract_full_text)
        self.biochem_values = pickle.load(open(os.path.join(args.input_dir, "biochem_values.pkl"), 'rb'))
        self.biochem_top = pickle.load(open(os.path.join(args.input_dir, "biochem_top.pkl"), 'rb'))
        self.query = f"""
SELECT pid, ts, data -> '{self.name}' as {self.name}
from jsontable where pid=%s and ts <= %s """
        self.admission_query = self.query + " and ts>%s"

    def apply_word_filter(self, word: str):
        if word!='pad':
            if self.resumed_params:
                word = get_npu_code(word,
                    top_biochem=self.resumed_params['top_biochem'], 
                    biochem_bins=self.resumed_params['biochem_bins'],       
                    include_percentile=self.resumed_params['include_percentile'], 
                    biochem_values=self.biochem_values, 
                    biochem_top=self.biochem_top)
            else:
                word = get_npu_code(word,
                    top_biochem=self.args.top_biochem, 
                    biochem_bins=self.args.biochem_bins,       
                    include_percentile=self.args.include_percentile, 
                    biochem_values=self.biochem_values, 
                    biochem_top=self.biochem_top)
                
        return word 
    
    def get_vocab(self) -> Vocab:
        parser = get_argument_parser() 
        generic_ds_vocab = os.path.join(self.args.input_dir, 'biochem_vocab.pkl')
        if self.resumed_params:
            vocab_path = os.path.join(self.args.base_dir, 
                self.ds_argument, 
                f"best_weights/{self.resumed_params['exp_id']}.vocab.{self.name}.pkl"
                )
            try:
                vocab = pickle.load(open(vocab_path, 'rb'))
                print (f"Vocab {self.name} log: Loaded vocab from {vocab_path}")
            except FileNotFoundError:
                print (f"""WARNING: vocab not found at {vocab_path}.
                This might mean all the parameters are default of there is a problem
                in the vocab exp id.""") #TODO add assert as in diag datasource
                vocab = pickle.load(open(generic_ds_vocab, 'rb'))
            return vocab

        if (parser.get_default("include_percentile") != self.args.include_percentile) or \
            (parser.get_default("top_biochem") != self.args.top_biochem) or \
            (parser.get_default("biochem_bins") != self.args.biochem_bins):
            custom_ds_vocab = f"best_weights/{self.args.exp_id}.vocab.{self.name}.pkl"
            if os.path.exists(custom_ds_vocab):
                vocab = pickle.load(open(custom_ds_vocab, 'rb'))
            else:
                vocab = generate_vocab_from_table("biochem", self.args.num_workers*2, self.args)
                with open(custom_ds_vocab, 'wb') as f:
                    pickle.dump(vocab, f)
        else:
            vocab = pickle.load(open(generic_ds_vocab, 'rb'))
        print ("Loaded Biochem vocab. Number of words :{}\n".format(vocab.n_words))
        return vocab

    def get2d_input(self, database_obj : MyDB, pid: str, time_limit: float):
        return super().get2d_input(database_obj, pid, time_limit)
