import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import numpy as np
from scipy import stats
import psycopg2
from sqlalchemy import create_engine

OUTPUT_PATH = 'output/2022-03-21-0414.ehr.tune/'
BEST_EXP_ID = '8ffc57282cb33791c148ba0316c03f8a'

#--------TRAINING LOSS ---------------#
cv = pd.read_csv(os.path.join(OUTPUT_PATH, 'CV_history_gridsearch.tsv'), sep='\t')
cv_best = cv[cv.exp_id == BEST_EXP_ID].iloc[:73][['epoch', 'loss', 'val_loss']]
long_cv_best= cv_best.melt(id_vars=['epoch'], value_vars=['loss', 'val_loss'])

fig = px.line(long_cv_best, x='epoch', y='value', color='variable', title='Training/Validation Loss')
fig.write_image('figures/pdig_revision/{}.training_loss.pdf'.format(BEST_EXP_ID))


#--------TIME TO EVENT ---------------#
results = pickle.load(open(os.path.join(OUTPUT_PATH, 'best_weights/{}_1.calibrated.test.pkl'.format(BEST_EXP_ID)), 'rb'))
results_df = pd.DataFrame(results)
results_df['real_tte'] = results_df.time_to_event - results_df.baselines
fig = px.histogram(
    results_df[(results_df.pred>.05)&(results_df.label==1.0)], 
    x='real_tte', 
    histnorm='probability', 
    nbins=100, 
    title='Time to event for positive samples',
    labels={'real_tte': 'Time to event (hours)'})
fig.write_image('figures/pdig_revision/{}.tte_pos.pdf'.format(BEST_EXP_ID))

fig = px.histogram(
    results_df[(results_df.pred>.05)&(results_df.label==1.0)], 
    x='real_tte', 
    histnorm='probability', 
    nbins=100, 
    title='Time to event for positive samples',
    cumulative=True,
    labels={'real_tte': 'Time to event (hours)'})
fig.write_image('figures/pdig_revision/{}.tte_pos_cum.pdf'.format(BEST_EXP_ID))


#-------- NOTES STATS ---------------#

query_notes_distribution = """
with tb1 as (
    select *, DATE(datetime) as dt from notestable
), tb2 as (
    select pid
        , dt 
        , count(*) as num_notes
    from tb1
    group by pid, dt
) select num_notes, count(*) as count_notes from tb2 group by num_notes order by num_notes
"""
engine_daplaci = create_engine('postgresql://daplaci@trans-db-01:5432/daplaci')
notes_dist = pd.read_sql(query_notes_distribution, con=engine_daplaci)
fig = px.bar(notes_dist, x='num_notes', y='count_notes', title='Number of notes per patient per day')
fig.write_image('figures/pdig_revision/supplementary.notes_dist.pdf'.format(BEST_EXP_ID))

query_notes_by_hour = """
with tb1 as (
    select pid
        , extract(hour from datetime) as hour_of_day 
    from notestable
) select hour_of_day
        , count(*) as count_notes
from tb1 
group by hour_of_day 
order by hour_of_day
"""

notes_by_hour = pd.read_sql(query_notes_by_hour, con=engine_daplaci)
fig = px.bar(notes_by_hour, x='hour_of_day', y='count_notes', title='Number of notes per hour of day')
fig.write_image('figures/pdig_revision/supplementary.notes_by_hour.pdf')

#-------- LABORATORY STATS ---------------#
query_labs_distribution = """
with tb1 as (
    select pid, 
        DATE(datetime) as dt, 
        jsonb_array_length(data->'biochem') as num_labs
    from jsontable
    where jsonb_array_length(data->'biochem') > 0
), tb2 as (
    select pid
        , dt 
        , sum(num_labs) as num_labs
    from tb1
    group by pid, dt
) select num_labs, count(*) as count_labs from tb2 group by num_labs order by num_labs
"""

labs_dist = pd.read_sql(query_labs_distribution, con=engine_daplaci)
fig = px.bar(labs_dist, x='num_labs', y='count_labs', title='Number of labs per patient per day')
fig.write_image('figures/pdig_revision/supplementary.labs_dist.pdf')

query_labs_by_hour = """
with tb1 as (
    select pid
        , extract(hour from datetime) as hour_of_day 
        , jsonb_array_length(data->'biochem')
    from jsontable
    where jsonb_array_length(data->'biochem') > 0
) select hour_of_day
        , count(*) as count_labs 
from tb1 
group by hour_of_day 
order by hour_of_day
"""

labs_by_hour = pd.read_sql(query_labs_by_hour, con=engine_daplaci)
fig = px.bar(labs_by_hour, x='hour_of_day', y='count_labs', title='Number of labs per hour of day')
fig.write_image('figures/pdig_revision/supplementary.labs_by_hour.pdf')
