#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 02:33:30 2023

@author: richardfremgen
"""

import pandas as pd
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt
from plotnine import *
from wordcloud import WordCloud
import os
import warnings
import pickle

warnings.filterwarnings('ignore')

os.chdir('/Users/richardfremgen/Documents/Portfolio/Code/Data')

#%% Define Helper Function

def clean_dict(data):
    
    """ Clean CV Hyperparatmer Data to only extract top performing model for 
        each tokenization configuration """
    
    for config, df in data.items():
        df['config'] = config

    out_df = pd.concat(sorted(data.values(), key=lambda df: df['config'][0]), ignore_index=True)
    out_df = out_df[out_df['rank_test_score'] == 1] 
    
    return(out_df.iloc[:, -14:])

 #%% Load in Cross Validation Pickle Files
 
pickle_in = open("m1_tune.pkl","rb")
m1_tune = pickle.load(pickle_in)
#del(pickle_in)

pickle_in = open("rf_tune.pkl","rb")
rf_tune = pickle.load(pickle_in)

pickle_in = open("xg_tune.pkl","rb")
xg_tune = pickle.load(pickle_in)
del(pickle_in)

# Preprocess m1_tune
mnb_tune = {k: v for k, v in m1_tune.items() if k.startswith('mnb')}
cnb_tune = {k: v for k, v in m1_tune.items() if k.startswith('cnb')}
knn_tune = {k: v for k, v in m1_tune.items() if k.startswith('knn')}
lsvc_tune = {k: v for k, v in m1_tune.items() if k.startswith('lsvc')}

#%% Run Clean Dict Function

xg_df = clean_dict(xg_tune)
rf_df = clean_dict(rf_tune)   
mnb_df = clean_dict(mnb_tune)
cnb_df = clean_dict(cnb_tune) 
knn_df = clean_dict(knn_tune)
knn_df = knn_df.drop_duplicates(subset = ['rank_test_score', 'config'])
lsvc_df = clean_dict(lsvc_tune) 

clean_df = pd.concat([xg_df, rf_df, mnb_df, cnb_df, lsvc_df, knn_df], ignore_index=True).reset_index(drop = True)
clean_df[['model', 'token']] = clean_df.config.str.split("_", 1, expand = True)
clean_df = clean_df.drop(['rank_test_score', 'mean_test_score', 'std_test_score'], axis = 1)

long_df = pd.melt(clean_df, 
                  id_vars = ['config','model', 'token'],
                  value_vars = clean_df.columns[:10].to_list())

long_df['token'] = long_df['token'].str.replace('bi_cv', 'BoW-Bi')
long_df['token'] = long_df['token'].str.replace('bi_tf', 'TFIDF-Bi')
long_df['token'] = long_df['token'].str.replace('combo_cv', 'BoW-Combo')
long_df['token'] = long_df['token'].str.replace('combo_tf', 'TFIDF-Combo')
long_df['token'] = long_df['token'].str.replace('uni_cv', 'BoW-Uni')
long_df['token'] = long_df['token'].str.replace('uni_tf', 'TFIDF-Uni')

long_df['model'] = long_df['model'].str.replace('cnb', 'Complement NB')
long_df['model'] = long_df['model'].str.replace('lsvc', 'Linear SVC')
long_df['model'] = long_df['model'].str.replace('mnb', 'Mulitnomial NB')
long_df['model'] = long_df['model'].str.replace('rf', 'Random Forest')
long_df['model'] = long_df['model'].str.replace('xg', 'XGBoost')
long_df['model'] = long_df['model'].str.replace('knn', 'K-Nearest Neighbor')

#%% Facet Grid 

colors = {'BoW-Bi':'white', 'TFIDF-Bi': 'lightblue', 'BoW-Combo' : 'lightgrey',
          'TFIDF-Combo':'red', 'BoW-Uni':'bisque', 'TFIDF-Uni' : 'firebrick'}  

(
    ggplot(long_df, aes(x='token', y='value', fill = 'token'))
    + geom_boxplot()
    + facet_wrap('model', ncol = 3)
    + scale_fill_manual(values = colors)
    + theme_bw()
    + labs(x = "Tokenization Method",
           y = "Fold Accuracy",
           title = "TF-IDF Uni & Comb. are the Best Performers",
           fill = "")
    + theme(
        plot_title = element_text(size=16, family="Arial", face="bold"), 
        text = element_text(colour="black", family="Arial"),
        axis_text_x = element_blank(),
        axis_ticks_major_x= element_blank(),
        axis_text_y = element_text(colour="black", family="Arial", face="bold", size=14))
    + theme(legend_position='bottom')
) 







