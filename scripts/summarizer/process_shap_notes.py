import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from unplanned_net.utilities.visualization import plot_vocab_datasource

icd_mapper_name = ""#path to icd mapper ICD -> description
icd_mapper = pd.read_csv(icd_mapper_name, sep='\t', 
                    names = ['icd10', 'definition', 'chapter_name', 'chapter', 'block_name', 'block'])
icd2definition = dict(zip(icd_mapper.icd10, icd_mapper.definition))
N = 3 

if  os.path.isfile('figures/grams{N}_shap_df.tsv'):
    shap_grams = pd.read_csv(f"figures/grams{N}_shap_df.tsv", sep='\t')
elif os.path.isfile("figures/shap_df.calibrated.tsv"):
    print ("Loading calibrated shap explanation ")
    shap_df = pd.read_csv("figures/shap_df.calibrated.tsv", sep='\t')
elif os.path.isfile("figures/shap_df.tsv"):
    print ("Loading NOT calibrated shap explanation ")
    shap_df = pd.read_csv("figures/shap_df.tsv", sep='\t')
else:
    raise FileExistsError

if __name__=="__main__":

    if "shap_grams" in globals():
        plot_vocab_datasource(shap_grams, np.array(['notes']), icd2definition, fig_suffix='ngrams', min_count=0)
    else:
        notes_shap = shap_df[shap_df.feat_type=='notes']
        notes_shap = notes_shap.groupby('adm').agg({'word':list, 'shap_value':list})

        records = []
        

        words2shap = zip(notes_shap.index, notes_shap.word, notes_shap.shap_value)
        grams = []
        grams_attributions = []
        adms = []
        for adm, sentence, shap_sentence in tqdm(words2shap):
            assert len(sentence) == len(shap_sentence)
            grams.extend([' '.join(map(str, sentence[i:i+N])) for i in range(len(sentence)-N+1)])
            grams_attributions.extend([sum(shap_sentence[i:i+N]) for i in range(len(shap_sentence)-N+1)])
            adms.extend([adm for i in range(len(shap_sentence)-N+1)])

        assert len(grams) == len(grams_attributions) == len(adms)

        shap_grams = pd.DataFrame({"adm":adms,"word":grams, "shap_value":grams_attributions})
        shap_grams["feat_type"] = "notes"
        shap_grams["feat_value"] = 1.0

        shap_grams.to_csv(f'figures/grams{N}_shap_df.tsv', index=False, sep='\t')


        plot_vocab_datasource(shap_grams, np.array(['notes']), icd2definition, fig_suffix='ngrams', min_count=0)
