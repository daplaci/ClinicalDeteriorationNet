rm(list=ls())

# import libraries
library("ggplot2")
library("reshape2")
library("dplyr")
library(reticulate)
np <- import("numpy")

# import data
setwd('#placeholder#')

auctable <- read.csv('AUC_history_gridsearch.tsv', header=TRUE, sep="\t", row.names = NULL)


group_vars <- setdiff(names(auctable), c('c_index', 'val_c_index', 'master_id',
                                         'exp_id', 'epoch', 'acc', 'val_acc', 
                                         'loss', 'val_loss', 
                                         'mcc', 'val_mcc', 
                                         'precision', 'val_precision',
                                         'recall', 'val_recall'))

grid_vars <- auctable[group_vars] %>%
  mutate_all(list(~length(unique(.)))) %>%
  distinct() %>%
  select_if(~(max(.) > 1))

auctable %>% 
  ggplot(aes(x=baseline_hours, y=val_mcc, color=binary_prediction)) +
  geom_point() + 
  geom_smooth()          

auctable %>% 
  mutate(
    units_category=as.factor(ntile(units,3))
  ) %>%  
  ggplot(aes(x=baseline_hours, y=val_mcc, color=units_category)) +
  geom_point() + 
  geom_smooth()          

auctable %>% 
  mutate(
    padd_biochem_category=as.factor(ntile(padd_biochem,3))
  ) %>%  
  ggplot(aes(x=baseline_hours, y=val_mcc, color=padd_biochem_category)) +
  geom_point() + 
  geom_smooth()

auctable %>% 
  mutate(
    padd_diag_category=as.factor(ntile(padd_diag,3))
  ) %>%  
  ggplot(aes(x=baseline_hours, y=val_mcc, color=padd_diag_category)) +
  geom_point() + 
  geom_smooth()          


auctable %>%
  pivot_longer(
    cols = c(age_at_adm, ews, idx_adm, time_windows)
  ) %>%   
  ggplot(aes(x=baseline_hours, y=val_mcc, color=value)) +
  facet_grid(~name) +
  geom_point() + 
  geom_smooth()  

auctable %>% 
  filter(
    baseline_hours == 72
  ) %>% 
  pivot_longer(
    cols = c(age_at_adm, ews, idx_adm, time_windows)
  ) %>%   
  ggplot(aes(x=padd_biochem, y=val_mcc, color=value)) +
  facet_grid(~name) +
  geom_point() + 
  geom_smooth()  

