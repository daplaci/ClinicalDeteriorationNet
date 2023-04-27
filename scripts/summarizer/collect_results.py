import json
import sys


assert sys.version_info > (3,5)
import captum.attr
import shlex
import os
import pandas as pd
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pickle
from collections import defaultdict
PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
sys.path.append(PATH_TO_LIB)
import unplanned_net.utilities.parser as parser
from unplanned_net.dataset.datasource import get_datasources, TextDataSource, NumericalDataSource
from unplanned_net.dataset.notes_datasource import NotesDataSource
import unplanned_net.dataset.data_utils as data_utils
import unplanned_net.run.model_utils as model_utils
import unplanned_net.run.learn as learn
from unplanned_net.utilities.html_writer import HtmlWriter
from unplanned_net.utilities.calibration import IsotonicLayer, CalibratedModel
from unplanned_net.utilities.vocab import *                                                     
from unplanned_net.utilities.collections import get_best_exp_optuna
from unplanned_net.utilities.visualization import plot_optuna, plot_attributions, plot_calibration, plot_catches


collect_args = argparse.ArgumentParser(description='parse args for performance analysis')
collect_args.add_argument('--collect_all', 
    action='store_true',
    default=False,
    help='If true, it will run over all the experiment in the folder')
collect_args.add_argument('--skip_eval', 
    action='store_true', 
    default=False,
    help='eval test')
collect_args.add_argument('--skip_attribution', 
    action='store_true',
    default=False, 
    help='generate attribution')
collect_args.add_argument('--apply_calibration', 
    action='store_true',
    default=False, 
    help='generate attribution')
collect_args.add_argument('--moab_path', 
    type=str, 
    help='path where to run this collection - if none use the running directory')
collect_args.add_argument('--truncate', 
    type=int,
    default=1000,
    help='Number of batches to use for attribution')
collect_args.add_argument('--attribution_algo', 
    type=str,
    default='shap',
    help='How to calculate the attribution')
collect_args.add_argument('--n_samples', 
    type=int,
    default=200,
    help='How to calculate the attribution')
collect_args.add_argument('--num_feat_plot', 
    type=int,
    default=30,
    help='how many features to show for both the direction of driving pred')
collect_args.add_argument('--plot_calibration', 
    action='store_true',
    default=False,
    help='if true plot calibration curves - as long as you have both the preds precomputed')
collect_args.add_argument('--plot_attribution', 
    action='store_true',
    default=False,
    help='if true plot atributions - as long as you have either the shap values saved or --skip_attribution False')
collect_args.add_argument('--overwrite', 
    action='store_true',
    default=False,
    help='if true overwrite eval and shap')

collect_args = collect_args.parse_args()

def eval_on_test(val_loader, model, optimizer, args, calibrate=False, age_std=1.0, age_mean=0.0, sex_std=1.0, sex_mean=0.0):
    """
        Load the best experiment and check performances on the test set.
    """
    val_gen = val_loader.__iter__()
    ## Save Evaluation on test set
    print ("Running evaluation on test.. \n")
    pred_list,label_list,baselines_list, age_at_adm_list, sex_list, time_to_event_list = [], [], [],[], [], []

    if calibrate:
        calibrate_fn = calibrate
    else:
        calibrate_fn = lambda x:x
    
    for batch in tqdm(val_gen):
        batch_results  = learn.val_batch(batch, model, args)
        pred_list.append(calibrate_fn(batch_results["output"]).cpu().data.numpy()) 
        label_list.append(batch_results["label"].cpu().data.numpy())
        baselines = batch_results["baselines"].cpu().data.numpy() - len(args.time_windows.split('-'))
        baselines_list.append(baselines*args.trigger_time)
        
        if "age_at_adm" in batch_results["inputs"]:
            ages = batch_results["inputs"]["age_at_adm"].cpu().data.numpy()
            ages = ages*age_std + age_mean
            age_at_adm_list.append(ages)
        if "sex" in batch_results["inputs"]:
            sexes = batch_results["inputs"]["sex"].cpu().data.numpy()
            sexes = sexes*sex_mean + sex_std
            sex_list.append(sexes)
        if 'time_to_event' in batch_results:
            time_to_event_list.append(batch_results["time_to_event"].cpu().data.numpy())

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)
    baselines = np.concatenate(baselines_list) if baselines_list else baselines_list
    age_at_adm = np.concatenate(age_at_adm_list) if age_at_adm_list else age_at_adm_list
    sex = np.concatenate(sex_list) if sex_list else sex_list
    time_to_event = np.concatenate(time_to_event_list) if time_to_event_list else time_to_event_list

    result = {'pred':pred, 'label':label, 'baselines':baselines, 'age_at_adm':age_at_adm, 'sex':sex, 'time_to_event':time_to_event}
    with open(test_filename, 'wb') as f:
        pickle.dump(result, f)

    return 0
    
def captum_explanation(train_loader, val_loader, model, optimizer, args, truncate=None, attribution_algo='shap', calibrate=False, n_samples=200):
    
    """
        Run Captum explanation: choose between shap or ig
    """
    torch.backends.cudnn.enabled=False
    print ("\Running {} \n".format(attribution_algo))
    assert attribution_algo in ['shap', 'ig']

    if calibrate:
        forward_fn = CalibratedModel(model.explainable_net, calibrate)
    else:
        forward_fn = model.explainable_net
    
    if attribution_algo == 'shap':
        explainer = captum.attr.GradientShap(forward_fn)
    elif attribution_algo == 'ig':
        explainer = captum.attr.IntegratedGradients(forward_fn)
    
    records = []
    val_gen = val_loader.__iter__()
    train_gen = train_loader.__iter__()
    pbar = tqdm(total=val_loader.__len__())
    def get_baseline():
        batch = next(train_gen)
        batch = learn.batch_to_device(batch, model.device)
        _, _, inputs, _, _,  _, _ = batch
        
        baseline_attributions = model.embeddings_net(inputs)
        return baseline_attributions

    batch_num = 0

    writer = HtmlWriter(f"{collect_args.attribution_algo}_text")
    try:
        while True:
            pbar.update(1)
            batch = next(val_gen)
            batch = learn.batch_to_device(batch, model.device)
            _, x_text, inputs, seq_lens, baselines,  label, _ = batch
        
            emb_inputs = model.embeddings_net(inputs)
            grad_attribution = explainer.attribute(
                                    emb_inputs, 
                                    baselines=get_baseline, 
                                    n_samples=n_samples,
                                    additional_forward_args=(seq_lens, baselines,),
                                    return_convergence_delta=False)

            preds = forward_fn(emb_inputs).squeeze().cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            for feat_n, ds in  enumerate(datasources):

                attribution_ft = grad_attribution[feat_n]
                attribution_ft = attribution_ft / torch.norm(attribution_ft)
                attribution_ft = attribution_ft.cpu().detach().numpy()

                if isinstance(ds, TextDataSource):
                    attribution_ft = attribution_ft.sum(-1)

                samples, *oth_dim = attribution_ft.shape

                if isinstance(ds, NotesDataSource):
                    notes_tkn = np.asarray(x_text['notes']).T
                
                for adm in range(samples):
                    shap_dict = defaultdict(lambda :0)
                    if label[adm]:
                        writer.write_html(f"sample -- {adm + batch_num*args.val_batch} label -- {label[adm]} pred -- {preds[adm]} baselines -- {baselines[adm]}", force_black=True)
                        writer.write_html(f"sample -- {adm + batch_num*args.val_batch} feature -- {ds.name}", force_black=True)
                    if isinstance(ds, TextDataSource):
                        words_dim, *_ = oth_dim
                        for w in range(words_dim):
                            if ds.name == "notes":
                                word = notes_tkn[adm, w]
                            else:
                                word = ds.vocab.index2word[int(inputs[ds.name][adm, w])]

                            if ds.name == 'diag':
                                try:
                                    word = icd2definition[word[1:]]
                                except:
                                    pass
                            if word not in ['unk', 'pad']:
                                if label[adm]: writer.write_html(word, float(attribution_ft[adm, w]))
                                shap_dict[word] += (float(attribution_ft[adm, w]))
                        
                    else:
                        if label[adm]: writer.write_html(f"{ds.name} -- {inputs[ds.name][adm]}", float(attribution_ft[adm]))
                        shap_dict[0] += float(attribution_ft[adm])
                    
                    for w in shap_dict:
                        if ds.vocab:
                            records.append((adm + batch_num*args.val_batch, ds.name, w, shap_dict[w], 1, preds[adm], label[adm]))
                        else:
                            feat_value = ds.stats['std']*(float(inputs[ds.name][adm]))+ds.stats['mean']
                            records.append((adm + batch_num*args.val_batch, ds.name, ds.name, shap_dict[w], feat_value, preds[adm], label[adm]))
            batch_num += 1
            if truncate and batch_num > truncate:
                break
    except StopIteration:
        pass

    columns_name = ["adm", "feat_type", "word", "shap_value", "feat_value", "pred", "label"]
    shapDF = pd.DataFrame.from_records(records, columns=columns_name)
    if calibrate:
        shap_name = 'figures/{}_df.calibrated.tsv'.format(attribution_algo) 
    else:
        shap_name = 'figures/{}_df.tsv'.format(attribution_algo) 
    shapDF.to_csv(shap_name, index=False, sep='\t')
    return shapDF

if __name__ == "__main__":

    if collect_args.moab_path:
        print (f"Collecting studies in {collect_args.moab_path}")
        os.chdir(collect_args.moab_path)

    icd_mapper_name = "" #define the path to the ICD mapper
    icd_mapper = pd.read_csv(icd_mapper_name, sep='\t', 
                        names = ['icd10', 'definition', 'chapter_name', 'chapter', 'block_name', 'block'])
    icd2definition = dict(zip(icd_mapper.icd10, icd_mapper.definition))
    icd2definition['R67'] = "Findings in assessing general functional ability"
    
    auctable = pd.read_csv('AUC_history_gridsearch.tsv', sep='\t')
    if collect_args.collect_all:
        best_exp_id = auctable.exp_id.unique().tolist()
    else:
        studies = set(auctable.study_id.tolist())
        if len(studies) > 1:
            raise NotImplementedError ("Only one study per grid search is currently implented in the evaluation")
        try:
            best_trials = get_best_exp_optuna(studies)
            best_exp_id = best_trials[0]#auctable.iptloc[best_trials[0].number - 1 ].exp_id
            for s in studies:
                plot_optuna(s)
        except Exception as e:
            print (f"{e}.\nError: could not load the study from the optnua storage.. reading it from the auctable")
            best_exp_id = auctable[auctable.val_auprc == max(auctable.val_auprc)].exp_id.tolist()[0]
        best_exp_id = [best_exp_id]

    for exp_id in best_exp_id:
        print ("Best exp id {}".format(exp_id))
        moab_filename = [f for f in os.listdir("moab_jobs/") if exp_id in f][0]
        flag_string = open("moab_jobs/{}".format(moab_filename), 'r').readlines()[-1]
        args = parser.parse_args(shlex.split(flag_string)[3:])
        args.train_batch = 16
        args.val_batch = 16
        args.verbose = True

        datasources = get_datasources(args)
        adm_data, ehr_data, t_person = data_utils.get_input_data(args)
        adm_test = json.load(open(os.path.join(args.input_dir, args.admissions_test),'r'))

        if collect_args.apply_calibration:
            fig_suffix= '.calibrated'
        else:
            fig_suffix=''    
        
        test_filename = 'best_weights/{}_1{}.test.pkl'.format(exp_id, fig_suffix)
        attribution_filename = 'figures/{}_df{}.tsv'.format(collect_args.attribution_algo, fig_suffix)
        file_written = os.path.isfile(test_filename) and not collect_args.overwrite
        skip_eval = collect_args.skip_eval or file_written
        
        if not (collect_args.skip_attribution and skip_eval):
            
            for p in sorted(adm_data):
                adm_data[p]['split_group'] = np.random.choice(['train','val'], p=[1.0, 0.0])

            train_dataset, train_loader = data_utils.get_dataclasses(adm_data, ehr_data, t_person, 'train', args, 
                                    datasources=datasources, 
                                    use_weights_sampler=True)

            for p in adm_test:
                adm_test[p]['split_group'] = 'test'

            val_dataset, val_loader = data_utils.get_dataclasses(adm_test, ehr_data, t_person, 'test', args, 
                                    datasources=datasources, 
                                    use_weights_sampler=False, 
                                    stats=train_dataset.stats)

            for ds in datasources:
                if isinstance(ds, NumericalDataSource):
                    ds.set_stats(train_dataset.stats)

            model, optimizer = model_utils.get_model(train_dataset.datasources, args,)
            state_dict = torch.load('best_weights/{}_1.pt'.format(args.exp_id))
            print ("Loading the best weights")
            model.load_state_dict(state_dict["model_state_dict"])
            model.eval()
        
            if collect_args.apply_calibration:
                #calibrate model
                val_preds = pickle.load(open('best_weights/{}_1.pkl'.format(args.exp_id), 'rb')) 
                isotonic_layer = IsotonicLayer(val_preds['pred'], val_preds['label'], device=model.device)
            else:
                isotonic_layer=None    
        
        if not skip_eval:
            eval_on_test(val_loader, model, optimizer, args, calibrate=isotonic_layer,
                age_mean=train_dataset.stats['age_at_adm']['mean'],
                age_std=train_dataset.stats['age_at_adm']['std'],
                sex_mean=train_dataset.stats['sex']['mean'],
                sex_std=train_dataset.stats['sex']['std'])

        #check if there are both files for calibration
        if collect_args.plot_calibration:
            calibrated_path = f"best_weights/{args.exp_id}_1.calibrated.test.pkl"
            uncalibrated_path = f"best_weights/{args.exp_id}_1.test.pkl"
            if os.path.exists(calibrated_path) and os.path.exists(uncalibrated_path):
                print ("Printing calibration curve")
                test_preds = pickle.load(open(calibrated_path, 'rb'))
                uncalibrated_test_preds = pickle.load(open(uncalibrated_path, 'rb'))
                plot_calibration(test_preds, uncalibrated_test_preds, exp_id=args.exp_id, n_boot=200)
                plot_catches(test_preds, exp_id=args.exp_id)
            else:
                raise FileNotFoundError(
                    "Either {} or {} files are not generated.".format(calibrated_path, uncalibrated_path))

        if not collect_args.skip_attribution:
            attributions = captum_explanation(train_loader, val_loader, model, optimizer, args, 
                truncate=collect_args.truncate, 
                attribution_algo=collect_args.attribution_algo, 
                calibrate=isotonic_layer, 
                n_samples=collect_args.n_samples)
        
        if collect_args.plot_attribution:
            if os.path.isfile(attribution_filename):
                print ("Found an attribution path previosly generated.. Loading and plotting")
                attributions = pd.read_csv(attribution_filename, sep='\t')
            else:
                raise FileNotFoundError("""Could not find any attribution.
                Run with without --skip_attribution or
                Check for error in attribution calculation.""")
            
            plot_attributions(attributions, datasources, icd2definition,
                fig_suffix=fig_suffix, 
                attribution_algo=collect_args.attribution_algo, 
                num_feat_plot=collect_args.num_feat_plot)
