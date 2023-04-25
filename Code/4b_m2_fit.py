#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:51:51 2023

@author: richardfremgen
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn import metrics 
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os
import warnings
import pickle 

warnings.filterwarnings('ignore')

os.chdir('/Users/richardfremgen/Documents/Portfolio/Code/Data')
df = pd.read_pickle("./df_clean_75.pkl") 

#%% Further Preprocess Text data

def lemmatize_words(text):
    
    """ Lemmatizes a data frame column """
    
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    
    return (' '.join(words)) 

def stem_sentences(sentence):
    
    """ Convert sentence to a stem for a data frame column """
    porter_stemmer = PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return (' '.join(stemmed_tokens))

def clean_data2(data):
    
    """ Clean and process data before performing sentiment analysis """ 
    
    #stop_words = stopwords.words('english') 
    #data['sentence'] = data['sentence'].apply(lemmatize_words)
    #data['sentence'] = data['sentence'].apply(remove_single_letter)
    data['sentence'] = data['sentence'].apply(stem_sentences) 
   
    return(data)

df = clean_data2(df) 

#%% Define Sentiment Analysis (SA) Function for Tree Based Ensemble Methods

def SA_Model(data, model_name, ngram_range = (1,1), model_type = RandomForestClassifier(), tf_idf = False, xg = False): 
    
    """ Run different configurations of Machine Learning Model"""
    
    if tf_idf == True: 
        
        # Preprocess data - TF-IDF Approach
        tfidf = TfidfVectorizer(ngram_range = ngram_range, binary=True) 
        text_counts = tfidf.fit_transform(data['sentence'])   
        
    else:     
        
        # Preprocess data - DTM Matrix
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(stop_words='english', ngram_range = ngram_range,  
                              tokenizer = token.tokenize, binary=True)
        text_counts = cv.fit_transform(data['sentence']) 
        
    if xg == True:
         
          data['sentiment'] = LabelEncoder().fit_transform(df['sentiment'])
        

    # Split into 64-16-20 train-validation-test sets
    X_train, X_test, y_train, y_test = train_test_split(text_counts, data['sentiment'], 
                                                        test_size=0.2, random_state=123) 
    
    model_fit = model_type.fit(X_train, y_train) 
    predictions = model_fit.predict(X_test) 
    
    # Get Classification Report
    # class_report = classification_report(y_test, predictions, digits=4)
    # print(class_report)
                    
    accuracy_score = round(metrics.accuracy_score(y_test,predictions),4)*100
    precision_score = round(metrics.precision_score(y_test, predictions, average= 'macro'), 4)
    recall_score = round(metrics.recall_score(y_test, predictions, average = 'macro'), 4)
    f1_score = round(metrics.f1_score(y_test, predictions, average= 'macro'), 4) 
           
    return_dict = {"model" : model_name, 
                    "acc" : accuracy_score, 
                    "precision" : precision_score,
                    "recall" : recall_score,
                    "f1_score" : f1_score} 
 
    return(pd.DataFrame([return_dict])) 

#%% Get Results for Ensemble Models 

def get_results(data): 
    
    """ Get Test Set Results from different ML Models """
    
    m1 = SA_Model(df, model_name = "RF CV U", ngram_range = (1,1), 
                  model_type = RandomForestClassifier(n_estimators= 150, random_state = 123), tf_idf = False)
    m2 = SA_Model(df, model_name = "RF CV B", ngram_range = (2,2), 
                  model_type = RandomForestClassifier(n_estimators= 100, random_state = 123), tf_idf = False)
    m3 = SA_Model(df, model_name = "RF CV UB", ngram_range = (1,2), 
                  model_type = RandomForestClassifier(n_estimators= 150, random_state = 123), tf_idf = False)
    m4 = SA_Model(df, model_name = "RF TFIDF U", ngram_range = (1,1), 
                  model_type = RandomForestClassifier(n_estimators= 225, random_state = 123), tf_idf = True) 
    m5 = SA_Model(df, model_name = "RF TFIDF B", ngram_range = (2,2), 
                  model_type = RandomForestClassifier(n_estimators= 225, random_state = 123), tf_idf = True)
    m6= SA_Model(df, model_name = "RF TFIDF UB", ngram_range = (1,2), 
                  model_type = RandomForestClassifier(n_estimators= 375, random_state = 123), tf_idf = True)
    
    m7 = SA_Model(df, model_name = "XG CV U", ngram_range = (1,1), 
                  model_type = XGBClassifier(n_estimators= 300, max_depth = 6, learning_rate =  0.25,
                                              objective = 'multi:softmax', random_state = 123), tf_idf = False, xg = True)
    
    m8 = SA_Model(df, model_name = "XG CV B", ngram_range = (2,2), 
                  model_type = XGBClassifier(n_estimators= 400, max_depth = 6, learning_rate =  0.25, 
                                              objective='multi:softmax',  random_state = 123), tf_idf = False, xg = True) 
    
    m9 = SA_Model(df, model_name = "XG CV UB", ngram_range = (1,2), 
                  model_type = XGBClassifier(n_estimators= 450, max_depth = 6, learning_rate =  0.2, 
                                              objective='multi:softmax',  random_state = 123), tf_idf = False, xg = True) 
    
    m10 = SA_Model(df, model_name = "XG TFIDF U", ngram_range = (1,1), 
                  model_type = XGBClassifier(n_estimators = 400, max_depth = 6, learning_rate =  0.2, 
                                              objective='multi:softmax', random_state = 123), tf_idf = True, xg = True) 
    
    m11 = SA_Model(df, model_name = "XG TFIDF B", ngram_range = (2,2), 
                  model_type = XGBClassifier(n_estimators= 150, max_depth = 9, learning_rate =  0.3, 
                                              objective='multi:softmax',  random_state = 123), tf_idf = True, xg = True)
    
    m12 = SA_Model(df, model_name = "XG TFIDF UB", ngram_range = (1,2), 
                  model_type = XGBClassifier(n_estimators= 450, max_depth = 6, learning_rate =  0.15, 
                                              objective='multi:softmax',  random_state = 123), tf_idf = True, xg = True) 
     
    m13 = SA_Model(df, model_name = "GBM CV U", ngram_range = (1,1), 
                  model_type = GradientBoostingClassifier(n_estimators= 250, max_depth = 9, learning_rate =  0.2, 
                                                          random_state = 123), tf_idf = False) 
    
    m14 = SA_Model(df, model_name = "GBM CV B", ngram_range = (2,2), 
                  model_type = GradientBoostingClassifier(n_estimators= 250, max_depth = 6, learning_rate =  0.3, 
                                                          random_state = 123), tf_idf = False) 
    
    m15 = SA_Model(df, model_name = "GBM CV UB", ngram_range = (1,2), 
                  model_type = GradientBoostingClassifier(n_estimators= 450, max_depth = 6, learning_rate =  0.3, 
                                                          random_state = 123), tf_idf = False) 
    
    m16 = SA_Model(df, model_name = "GBM TFIDF U", ngram_range = (1,1), 
                  model_type = GradientBoostingClassifier(n_estimators = 450, max_depth = 6, learning_rate =  0.2, 
                                                          random_state = 123), tf_idf = True) 
    
    m17 = SA_Model(df, model_name = "GBM TFIDF B", ngram_range = (2,2), 
                  model_type = GradientBoostingClassifier(n_estimators= 300, max_depth = 9, learning_rate =  0.2, 
                                                          random_state = 123), tf_idf = True)
    
    m18 = SA_Model(df, model_name = "GBM TFIDF UB", ngram_range = (1,2), 
                  model_type = GradientBoostingClassifier(n_estimators= 450, max_depth = 6, learning_rate =  0.3, 
                                                          random_state = 123), tf_idf = True) 
            
    m_df = pd.concat([m1, m2, m3, m4, m5, m6, m7, m8, m9,
                      m10, m11, m12, m13, m14, m15, m16, m17, m18], ignore_index=True) 
    
        
    return(m_df) 
 
print("\n") 
print("=================== TEST SET RESULTS ===================")
print(get_results(df)) 


