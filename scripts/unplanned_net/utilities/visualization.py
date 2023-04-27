import optuna
import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_curve
import sys

PATH_TO_LIB = "/users/projects/clinical_deterioration/scripts"
sys.path.append(PATH_TO_LIB)

figname_to_ds = {"E - Medical Notes":"notes",
"C - Biochemical measurements":"biochem",
"A - Diagnoses":"diag",
"B - Age at admission":"age_at_adm",
"F - Sex":"sex",
"D - Number of previous admissions":"idx_adm"}

ds_to_figname = {v:k for k,v in figname_to_ds.items()}

def insert_linebreak(line, every=10):
    return '\n'.join(line[i:i+every] for i in range(0, len(line), every))

apply_sigmoid = lambda x :  1 / (1+ np.exp(-x))

def plot_optuna(study_id):

    from unplanned_net.utilities.collections import get_successfull_trials
    study = optuna.create_study()
    for t in get_successfull_trials(study_id):
        study.add_trial(t)
    optuna.visualization.matplotlib.plot_param_importances(study)
    ax = plt.gca()
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabels = [el.replace('_', ' ').capitalize() for el in ylabels]
    ax.set_yticklabels(ylabels)
    plt.savefig(f'figures/study.{study_id}.importances.svg',format='svg', dpi=300, bbox_inches='tight') 
    plt.savefig(f'figures/study.{study_id}.importances.pdf',format='pdf', dpi=300, bbox_inches='tight') 
    optuna.visualization.matplotlib.plot_slice(study)
    f = plt.gcf()
    for ax in f.axes:
        xlabel = ax.get_xlabel()
        xlabel = xlabel.replace('_', ' ').capitalize()
        ax.set_xlabel(xlabel)
    plt.savefig(f'figures/study.{study_id}.slices.svg',format='svg', dpi=300, bbox_inches='tight') 
    plt.close()
    return 0

def my_palplot(pal, size=1, ax=None, ylabel=None):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot
    ax :
        an existing axes to use
    """

    n = len(pal)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(n,1),
              cmap=matplotlib.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.get_xaxis().set_visible(False)
    ax.set_yticks(range(n))
    ax.set_yticklabels((np.linspace(ylabel[0], ylabel[-1], n, dtype=np.int16)))

def plot_vocab_datasource(shap_long, ds, icd2definition, 
                          fig_suffix='', attribution_algo='shap', 
                          num_feat_plot=10, min_count=50, ax=None):

    plot_name = ds_to_figname[ds]
    shap_long["word"] = shap_long["word"].apply(
        lambda x:(icd2definition[x[1:]] if type(x) is str and x[1:] in icd2definition else x)
        )
    
    ds_shap = shap_long.loc[shap_long.feat_type==ds, ['adm', 'word', 'shap_value', 'feat_value']]
    assert len(ds_shap)
    ds_shap = ds_shap[ds_shap.word!='pad']
    columns = ds_shap.word.unique().tolist()
    
    def filter (x):
        if countw.loc[x.name]>min_count:
            return x['shap_value']/num_admissions #this returned the averaged shap including the fact the some admissions have no 
            #return x['shap_value']/countw.loc[x.name] #this just average over the shap positive (doesnt care how often the disease was assigned)
        else:
            return 0
            
    grouped = ds_shap.groupby(['word']).sum() #apply(lambda x:np.abs(x) if x.name=='TODO' else x)
    countw = ds_shap.groupby('word')['shap_value'].count()
    num_admissions = len(ds_shap.adm.unique())
    grouped = grouped.apply(filter, axis=1)
    grouped = grouped.sort_values()
    N_FEAT = num_feat_plot//2

    if N_FEAT < len(columns):
        topk = grouped.iloc[-N_FEAT:].index.tolist()[::-1] + grouped.iloc[:N_FEAT].index.tolist()
    else:
        topk = grouped.index.tolist()
    
    if not ax:
        fig, ax = plt.subplots(figsize=(15, 11))
    else:
        fig = None

    plt_data = ds_shap[ds_shap.word.isin(topk)]
    zscores = plt_data.groupby('word').shap_value.transform(lambda x:zscore(x))
    zscores =zscores.fillna(0)
    plt_data = plt_data.loc[np.abs(zscores)<3, :]

    if ds in ['idx_adm', 'age_at_adm', 'sex']:
        plt_data['bin'] = pd.qcut(plt_data['feat_value'], 10, duplicates='drop').apply(
                lambda x : "{}-{}".format(x.left, x.right))
        plt_data['shap_bin'] = pd.cut(plt_data['shap_value'], 50).apply(
                lambda x : "{}-{}".format(x.left, x.right))
        
        n_colors = len(plt_data.bin.unique().categories.values)
        n_colors = max(n_colors, 2)

        plt_data['y'] = 0
        counter = plt_data.groupby('shap_bin').agg({'y':'count'}).reset_index()
        plt_data = plt_data.merge(counter, on='shap_bin')
        plt_data['y_y'] = plt_data['y_y']*np.random.randn(plt_data.shape[0])
        
        if n_colors > 2:
            hue_col = 'bin'
        else:
            hue_col='feat_value'
        palette = sns.diverging_palette(220,20,n=n_colors)
        sns.scatterplot(data=plt_data, x='shap_value', y='y_y', 
            hue=hue_col, palette=palette, ax=ax)
        ax.set(ylabel=plot_name, xlabel='Attribution')
        ax.axes.yaxis.set_ticks([])
        axin = ax.inset_axes([0.9,0.55,0.05,0.3])
        
        axin_yticks = plt_data.bin.unique().categories.values
        axin_yticks = [subel for el in axin_yticks for subel in el.split("-")]
        axin_yticks = [round(float(el)) for el in axin_yticks if el.replace('.','').isdigit()]
        axin_yticks = list(dict.fromkeys(axin_yticks))
        my_palplot(palette, ax=axin, ylabel=axin_yticks)
        ax.legend().remove()
    else:
        sns.boxplot(x="shap_value", y="word", data=plt_data, width=.6, 
            palette="vlag_r", whis = 4, fliersize=.5, order=topk, ax=ax)
        sns.stripplot(x="shap_value", y="word", data=plt_data.sample(frac=0.05),
            size=4, color=".3", linewidth=0, order=topk, alpha=0.5, ax=ax)
    
    
        yticks = [a.get_text() for a in ax.get_yticklabels()]
        yticks = [el.replace("@", ' ').replace("_P_", " Plasma ").replace("_B_", " Blood ").capitalize() for el in yticks]
        yticks = [el  if el!='X' else "Findings in assessing general functional ability" for el in yticks]
        
        yticks = [insert_linebreak(el, every=50) for el in yticks ]
        ax.set_yticklabels(yticks)
    
    ax.xaxis.grid(True)
    ax.set(ylabel="", xlabel="Attribution", xlim=(-1,1))
    ax.set_title(f'{plot_name}', fontdict={'fontsize':18, 'fontweight': 'medium'})
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if fig is not None:
        plt.savefig(f"figures/{attribution_algo}_boxplot_summary_{fig_suffix}_{ds}.svg", 
            bbox_inches='tight', format='svg')
    
def plot_attributions(shapDF, datasources, icd2definition, fig_suffix='', attribution_algo='shap', num_feat_plot=10):

    shap_long = shapDF.loc[:, [c not in ["pred", "label"] for c in shapDF.columns]]
    
    r = len(datasources)//2 + len(datasources)%2
    c = min(2, len(datasources))
    _, axes = plt.subplots(r,c, figsize=[18,38], constrained_layout=True)
    
    index_fig = 0
    for figname in sorted(figname_to_ds):
        ds = figname_to_ds[figname]
        if ds not in [d.name for d in datasources]:
            continue
        plot_vocab_datasource(
            shap_long,
            ds,
            icd2definition,
            fig_suffix=fig_suffix, 
            attribution_algo=attribution_algo,
            num_feat_plot=num_feat_plot, 
            ax=None)#axes[index_fig//2, index_fig%2]
        index_fig += 1
        
        
    save_figure_and_subplots(f"figures/{attribution_algo}_boxplot_summary_{fig_suffix}.svg", 
        plt.gcf(), dpi=300, format='svg')
    plt.close()

def regression_coordinates(x, y):
    #y = ax + b
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    a_num = np.sum((x - x_mean) * (y - y_mean))
    a_den = np.sum((x - x_mean)**2)
    a = a_num / a_den
    
    b = y_mean - (a*x_mean)
    return a, b

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    hm, m, mh = np.percentile(a, (2.5, 50, 97.5))
    return np.around(m, decimals=3), np.around(hm, decimals=3), np.around(mh, decimals=3)

def calibration_plot(preds, labels, x=None, y=None, n_boot=1000, 
                     ax=None, xlim=(0,1), ylim=(0,1), label=None):
    
    if np.any(preds <0):
        preds = apply_sigmoid(preds).round(3).squeeze()
        if isinstance(x, np.ndarray):
            x = apply_sigmoid(x).round(3)
    
    preds = preds.round(3)
    records = []
    reg_coord = []
    unique_probs = np.unique(preds)
    for _ in range(n_boot):
        index = np.random.choice(len(preds), len(preds), replace=True)
        _preds = preds[index]
        _labels = labels[index]
        boot_calib_coord = []
        for p in unique_probs:
            positive_preds = _preds == p# test '>'
            if np.any(positive_preds):
                _labels_positive_preds = _labels[positive_preds]
                percentage = sum(_labels_positive_preds)/len(_labels_positive_preds)
            else:
                percentage = 0
            boot_calib_coord.append((p, percentage))
        records.extend(boot_calib_coord)
        x_, y_ = zip(*boot_calib_coord)
        reg_coord.append(regression_coordinates(x_, y_))
    a, b = zip(*reg_coord)

    slope_ci = mean_confidence_interval(a)
    intercept_ci = mean_confidence_interval(b) 
    print ("Calibration slope : {}\nCalibration intercept : {}".format(
        slope_ci,
        intercept_ci
    ))
    
    df = pd.DataFrame.from_records(records, columns=['probs', 'percentage'])
    sns.lineplot(x='probs', y='percentage', data=df, err_style="bars", ci=95, ax=ax, label=label)
    plt.plot(np.linspace(*xlim,100),np.linspace(*ylim,100), alpha=.3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    return slope_ci, intercept_ci

def plot_calibration(test_preds, uncalibrated_test_preds, n_boot=1, exp_id=''):
    _, ax = plt.subplots(1, 1, figsize=[6,6])
    slope, intercept = calibration_plot(test_preds['pred'], test_preds['label'], n_boot=n_boot, ax=ax, label='Calibrated Prediction')
    ax.text(0.7 ,0.5, 'Calibrated Slope {}\nCalibrated Intercept {}'.format(slope, intercept),)
    slope, intercept = calibration_plot(uncalibrated_test_preds['pred'], uncalibrated_test_preds['label'], n_boot=n_boot, ax=ax, label='Raw Prediction')
    ax.text(0.7 ,0.2, 'Slope {}\nIntercept {}'.format(slope, intercept),)
    ax.set(title="Calibration plot", xlabel='Probability', ylabel="Percentage")
    plt.savefig('figures/{}.calibration_plot.svg'.format(exp_id), bbox_inches='tight', format='svg')
    return 0

def plot_catches(test_preds, exp_id):
    precision, recall, thresholds = precision_recall_curve(test_preds["label"], test_preds["pred"])

    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix]
    early_catch = [t-b for t,b,p,l in zip(test_preds['time_to_event'],
                                    test_preds['baselines'],
                                    test_preds['pred'],
                                    test_preds['label'])
                                    if p>best_threshold and l==1]
    
    fig, ax = plt.subplots(1, 1, figsize=[6,6])
    sns.displot(early_catch, ax=ax)
    sns.displot(early_catch, kind='ecdf', ax=ax)
    ax.set(title="True positive catches to Event", ylabel="Density")
    plt.savefig('figures/{}.catches.svg'.format(exp_id), bbox_inches='tight', format='svg')
    return 0

def save_figure_and_subplots(figname, fig, **kwargs):
    """
        figname: name of the file to save
        fig: matplotlib.pyplot.figure object

        This function takes some figure and save its content. In addition it save all its axes/subfigure
        for offline processing.
        
    """
    if 'format' not in kwargs:
        kwargs['format'] = 'png'
    for ax_num,ax in enumerate(fig.axes):
        extent = ax.get_tightbbox(renderer=fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("{}_{}.{}".format(figname, ax_num, kwargs['format']), bbox_inches=extent, **kwargs)
    fig.savefig("{}.{}".format(figname, kwargs['format']), bbox_inches='tight', **kwargs)