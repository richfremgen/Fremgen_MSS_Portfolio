#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:26:01 2023

@author: richardfremgen
"""

import pandas as pd
import numpy as np
import os
import json 

# warnings.filterwarnings('ignore')

os.chdir('/Users/richardfremgen/Documents/Portfolio/Code/Data')

#%% Define Text Cleaning and Data Import Function

def get_json_data(data, source):
    
    """ Converts SemEval json file into a tidy data frame conducive for analysis """
   
    save_sentence = []
    save_score = []
    
    # Load JSON files to tidy
    with open(data) as json_file: 
        data_dict = json.load(json_file)
    
    for sent in data_dict: 
        save_sentence.append(sent['title'])  
        save_score.append(sent['sentiment'])
        
    clean_df = pd.DataFrame({'text' : save_sentence,
                           'specific_score' : save_score})
    
    clean_df['specific_score'] = pd.to_numeric(clean_df['specific_score'])
    clean_df['sent_score'] = np.where(clean_df['specific_score'] > 0, 1,
                                 np.where(clean_df['specific_score'] == 0, 0, -1))
    
    clean_df['sentiment'] = np.where(clean_df['specific_score'] > 0, "positive",
                                 np.where(clean_df['specific_score'] == 0, "neutral", "negative"))
    clean_df['source'] = source 
    clean_df['sentence'] = clean_df['text']
    clean_df = clean_df[['text','sentence', 'sentiment', 'sent_score', 'source']]
    
    return(clean_df)

def remove_single_letter(text):
    
    """ Remove words of length one from training corpus """
    
    words = text.split()
    new_list = [str for str in words if len(str) > 1]
    
    return(' '.join(new_list))  

def clean_data(data) :
    
    """ Removes unwanted phrases and symbols from sentence column """
    
    # Remove unnecessary text from labeled sentences
    data['sentence'] = data['sentence'].str.lower()
    data['sentence'] = data['sentence'].str.replace("\([^()]*\)", "", regex = True) 
    data['sentence'] = data['sentence'].str.replace(r"\bjan\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bfeb\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bmar\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bapr\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bmay\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bjun\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bjul\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\baug\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bsep\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\boct\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bnov\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bdec\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bjanuary\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bfebruary\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bmarch\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bapril\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bmay\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bjune\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bjuly\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\baugust\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bseptember\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\boctober\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bnovember\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bdecember\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace('[^a-zA-Z]', ' ', regex=True) 
    data['sentence'] = data['sentence'].str.replace("\s+", " ", regex=True).str.strip()
    data['sentence'] = data['sentence'].str.replace(r"\bm\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\bmn\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bmillion\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\beur\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bbillion\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bpct\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bk\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bbln\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bpercent\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bfinnish\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\beuro\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bfinland\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\bhelsinki\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\boyj\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace("\s+", " ", regex=True)
    data['sentence'] = data['sentence'].apply(remove_single_letter)
    
    return(data) 

#%%

# Load Financial Phrasebank Data 
data_fpb = pd.read_csv('Sentences_75Agree.txt', encoding = "ISO-8859-1",  
                       names=['text','sentiment'], delimiter= '@')

# Preprocess Financial Phrase Bank Data
data_fpb = data_fpb.drop_duplicates(subset = ['text']) 
#data_fpb.at[3012, 'text'] = data_fpb.at[3012, 'text'].replace("(", '').replace(")", '').strip() 
data_fpb['sentence'] = data_fpb['text']
data_fpb['sent_score'] = np.where(data_fpb['sentiment'] == 'positive', 1,
                             np.where(data_fpb['sentiment'] == 'neutral', 0, -1))
data_fpb['source'] = 'fpb' 
data_fpb = data_fpb[['text', 'sentence', 'sentiment', 'sent_score', 'source']] 
 
# Load SemEval data
data_semeval = get_json_data(data = "SemEval_Task5.2.json", source = "semeval") 
data_semeval = data_semeval.drop_duplicates(subset = ['text'])
    
#Combine Data
df_clean = pd.concat([data_fpb, data_semeval], ignore_index=True)
df_clean = clean_data(df_clean)

# Export df_clean to pickle file in working directory
# df_clean.to_pickle("./df_clean_75.pkl") 



 

  