#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:51:51 2023

@author: richardfremgen
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import metrics 
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
#from lightgbm import LGBMClassifier
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os
import time
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

#%% Tune Model Function

def Tune_Model(data, model_type, param_grid, n_iter, ngram_range = (1,1), tf_idf = False, search_type = 'grid', xg = False):  
    
    """ Find optimal hypermatters """ 
    
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
        
    # Split into 80-20 train-validation-test sets 
    X_train, X_test, y_train, y_test = train_test_split(text_counts, data['sentiment'], 
                                                        test_size=0.2, random_state=123) 
    
    tune_model = model_type 
    
    if search_type == "grid":
        
        clf = GridSearchCV(tune_model, param_grid = param_grid, cv = 10, 
                           scoring='accuracy', n_jobs = -1)
        best_clf = clf.fit(X_train,y_train)
        print("Tuned Hyperparameters :", best_clf.best_params_)
        print("Accuracy :",best_clf.best_score_)
        
    else: 
        
        clf = RandomizedSearchCV(tune_model, param_distributions = param_grid, 
                                 n_iter = n_iter, cv = 10, scoring='accuracy', n_jobs = -1) 
        best_clf = clf.fit(X_train,y_train)
        print("Tuned Hyperparameters :", best_clf.best_params_)
        print("Accuracy :",best_clf.best_score_)
        
    # Return the results from each split 
    test_df = pd.DataFrame(best_clf.cv_results_)
    
    return(test_df)

#%% Find optimal paramters

start = time.time()

# Tune Random Forest Model

param_grid = [{'n_estimators' : list(range(100, 525, 25))}]

rf_uni_cv = Tune_Model(df, model_type = RandomForestClassifier(random_state = 123), param_grid = param_grid, 
                          n_iter = 20, ngram_range = (1,1), tf_idf = False, search_type = "grid")  

rf_bi_cv = Tune_Model(df, model_type = RandomForestClassifier(random_state = 123), param_grid = param_grid, 
                          n_iter = 20, ngram_range = (2,2), tf_idf = False, search_type = "grid")  

rf_combo_cv = Tune_Model(df, model_type = RandomForestClassifier(random_state = 123), param_grid = param_grid, 
                          n_iter = 20, ngram_range = (1,2), tf_idf = False, search_type = "grid") 

rf_uni_tf = Tune_Model(df, model_type = RandomForestClassifier(random_state = 123), param_grid = param_grid, 
                          n_iter = 20, ngram_range = (1,1), tf_idf = True, search_type = "grid")  

rf_bi_tf = Tune_Model(df, model_type = RandomForestClassifier(random_state = 123), param_grid = param_grid, 
                          n_iter = 20, ngram_range = (2,2), tf_idf = True, search_type = "grid") 

rf_combo_tf = Tune_Model(df, model_type = RandomForestClassifier(random_state = 123), param_grid = param_grid, 
                          n_iter = 20, ngram_range = (1,2), tf_idf = True, search_type = "grid") 

# Tune XG Boost Model

# param_grid = [{'n_estimators' : list(range(100, 550, 50)),
#                'max_depth' : [3, 6, 9],
#                'learning_rate' : list(np.linspace(0.1, 0.3, 11),
#                'objective' : ['multi:softprob'] }] 

param_grid = [{'n_estimators' : list(range(150, 500, 50)), 
                'max_depth' : [3, 6, 9],
                'learning_rate' : [0.1, 0.2, 0.3, 0.15, 0.25] ,
                'objective' : ['multi:softprob'] }]

xg_uni_cv  = Tune_Model(df, model_type = XGBClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,1), tf_idf = False, search_type = "grid", xg = True)  

xg_bi_cv  = Tune_Model(df, model_type = XGBClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (2,2), tf_idf = False, search_type = "grid", xg = True) 

xg_combo_cv  = Tune_Model(df, model_type = XGBClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,2), tf_idf = False, search_type = "grid", xg = True)  

xg_uni_tf  = Tune_Model(df, model_type = XGBClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,1), tf_idf = True, search_type = "grid", xg = True)  

xg_bi_tf  = Tune_Model(df, model_type = XGBClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (2,2), tf_idf = True, search_type = "grid", xg = True)  

xg_combo_tf  = Tune_Model(df, model_type = XGBClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,2), tf_idf = True, search_type = "grid", xg = True) 

# Tune Gradient Boosting Machine  
 
param_grid = [{'n_estimators' : [150, 200, 250, 300, 450, 500],  
                'max_depth' : [3, 6, 9],
                'learning_rate' : [0.1, 0.2, 0.3] }] 

gbm_uni_cv  = Tune_Model(df, model_type = GradientBoostingClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,1), tf_idf = False, search_type = "random")  

gbm_bi_cv  = Tune_Model(df, model_type = GradientBoostingClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (2,2), tf_idf = False, search_type = "random") 

gbm_combo_cv  = Tune_Model(df, model_type = GradientBoostingClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,2), tf_idf = False, search_type = "random")  

gbm_uni_tf  = Tune_Model(df, model_type = GradientBoostingClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,1), tf_idf = True, search_type = "random")  

gbm_bi_tf  = Tune_Model(df, model_type = GradientBoostingClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (2,2), tf_idf = True, search_type = "random")  

gbm_combo_tf  = Tune_Model(df, model_type = GradientBoostingClassifier(random_state = 123), param_grid = param_grid, 
            n_iter = 20, ngram_range = (1,2), tf_idf = True, search_type = "random") 


print(f"Total HP Time: {round((time.time()-start)/60,2)} min") 

#%%

def top_hp(data, name, model = 'rf') :
    
    """ Extract top performing hyperparamter from ML validation sets """
    
    if model == 'rf':
        
        col_save = ['model', 'param_n_estimators', 'mean_test_score', 'std_test_score']
        best_p = data[data['rank_test_score'] == 1]
        best_p['model'] = name
        best_p = best_p[col_save]
                
    return(best_p)

#Random Forest - HP
cv1 = top_hp(rf_uni_cv, name = "rf_uni_cv", model = "rf")
cv2 = top_hp(rf_bi_cv, name = "rf_bi_cv", model = "rf")
cv3 = top_hp(rf_combo_cv, name = "rf_combo_cv", model = "rf")
cv4 = top_hp(rf_uni_tf, name = "rf_uni_tf", model = "rf")
cv5 = top_hp(rf_bi_tf, name = "rf_bi_tf", model = "rf")
cv6 = top_hp(rf_combo_tf, name = "rf_combo_tf", model = "rf")

cv_df_rf = pd.concat([cv1, cv2, cv3, cv4, cv5, cv6], ignore_index=True)

del cv1, cv2, cv3, cv4, cv5, cv6 

print("\n")
print("=================== RF HYPERPARAMTER RESULTS ===================")
print(cv_df_rf)

#%% Save Cross Validation Results and Export to Pickle File

#Save CV Results to one dicitonary  
rf_tune =  {"rf_uni_cv" : rf_uni_cv, "rf_bi_cv" : rf_bi_cv, "rf_combo_cv" : rf_combo_cv, 
            "rf_uni_tf" : rf_uni_tf, "rf_bi_tf" : rf_bi_tf, "rf_combo_tf" : rf_combo_tf}

xg_tune =  {"xg_uni_cv" : xg_uni_cv, "xg_bi_cv" : xg_bi_cv, "xg_combo_cv" : xg_combo_cv, 
            "xg_uni_tf" : xg_uni_tf, "xg_bi_tf" : xg_bi_tf, "xg_combo_tf" : xg_combo_tf}

gbm_tune =  {"gbm_uni_cv" : gbm_uni_cv, "gbm_bi_cv" : gbm_bi_cv, "gbm_combo_cv" : gbm_combo_cv, 
            "gbm_uni_tf" : gbm_uni_tf, "gbm_bi_tf" : gbm_bi_tf, "gbm_combo_tf" : gbm_combo_tf}


# # Export to Pickle File 
# file_to_write = open("gbm_tune.pkl", "wb")
# pickle.dump(gbm_tune, file_to_write)  

#%% Load Pickle File - CV Results
# import pickle
# pickle_in = open("gbm_tune.pkl","rb")
# gbm_df = pickle.load(pickle_in)
# del(pickle_in)