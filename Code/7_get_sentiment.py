#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:50:15 2023

@author: richardfremgen
"""

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer, LancasterStemmer
import numpy as np
from nltk.tokenize import sent_tokenize
import joblib
import pandas as pd
import pickle
import os

os.chdir('/Users/richardfremgen/Documents/Portfolio/Code/Data')

# Load API Data 
df_articles = pd.read_pickle("./api_articles.pkl") 
df = df_articles[['summary']]
df = df.rename(columns = {'summary' : 'sentence'})

#%% Preprocessing Helper Functions

def remove_single_letter(text):
    
    """ Remove words of length one from training corpus """
    
    words = text.split()
    new_list = [str for str in words if len(str) > 1]
    
    return(' '.join(new_list))  

def stem_sentences(sentence):
    
    """ Convert sentence to a stem for a data frame column """
    porter_stemmer = PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return (' '.join(stemmed_tokens))

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
    data['sentence'] = data['sentence'].str.replace(r"\bduke\b", " ", regex=True) 
    data['sentence'] = data['sentence'].str.replace(r"\benergy\b", " ", regex=True)
    data['sentence'] = data['sentence'].str.replace("\s+", " ", regex=True)
    data['sentence'] = data['sentence'].apply(remove_single_letter)
    #data['sentence'] = data['sentence'].apply(stem_sentences) 
    
    return(data)  

def get_sentiment(data, model, big_df):
    
    """ Predict sentiment of API article data """
    
    # Only save specific columns 
    col_save = ['title', 'published_date', 'clean_url', 'summary', 
                'rank', '_score', '_id', 'sentiment']
    
    # Get Predictions 
    X_new_test =  data['sentence'].to_list()         
    predictions = model.predict(X_new_test) 
    
    # Append sentiment to df and clean data 
    big_df['sentiment'] = predictions    
    big_df = big_df[col_save]
    big_df = big_df.rename(columns = {'_score' : 'score',
                                      'clean_url' : 'source'})

    return(big_df)

#%% Load Final Linear SVC Model and Get Sentiment
                
# Load Final Model
fittedModel = joblib.load('final_model.pkl')


# Run Clean Data Fun to Prep for Analysis and Get Sentiment Scores
df = clean_data(df)
sent_df = get_sentiment(df, model = fittedModel, big_df=df_articles)  


# sent_df.to_excel('sent_df.xlsx', index = False)

#%% Compute TF-IDF Scores 
tab_df = sent_df[['summary', 'sentiment']]
tab_df = tab_df.rename(columns = {'summary' : 'sentence'})
tab_df = clean_data(tab_df)

# df_neg = tab_df[tab_df['sentiment'] == 'Negative']
# df_pos = tab_df[tab_df['sentiment'] == 'Positive']
# df_neu = tab_df[tab_df['sentiment'] == 'Neutral']

# neg_vec = ''.join(df_neg['sentence'].tolist())
# pos_vec = ''.join(df_pos['sentence'].tolist())
# neu_vec = ''.join(df_neu['sentence'].tolist())

# word_list = [neg_vec, pos_vec, neu_vec]
# sent_list = ['Negative', 'Positive', 'Neutral']
# word_df = pd.DataFrame({'Sentiment' : sent_list, 'Words' : word_list})

#%%

tfidf_vectorizer = TfidfVectorizer(ngram_range=[1, 2], stop_words='english')
tfidf_separate = tfidf_vectorizer.fit_transform(tab_df['sentence'])
df_tfidf = pd.DataFrame(tfidf_separate.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), 
                        index=tab_df.index)

long_df = df_tfidf.T
