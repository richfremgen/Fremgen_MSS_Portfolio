#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:52:57 2023

@author: richardfremgen
"""

from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer, LancasterStemmer
import pickle
import os
import warnings

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

def Tune_Model(data, model_type, param_grid, ngram_range = (1,1), tf_idf = False):  
    
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
        

    # Split into 80-20 train-validation-test sets 
    X_train, X_test, y_train, y_test = train_test_split(text_counts, data['sentiment'], 
                                                        test_size=0.2, random_state=123) 
    
    tune_model = model_type 
    
    clf = GridSearchCV(tune_model, param_grid = param_grid, cv = 10, 
                       scoring='accuracy', n_jobs = -1)

    best_clf = clf.fit(X_train,y_train)
    print("Tuned Hyperparameters :", best_clf.best_params_)
    print("Accuracy :",best_clf.best_score_)
    
    # If you want to return the results from every split
    test_df = pd.DataFrame(best_clf.cv_results_)
    return(test_df)

#%% Find Optimal Hyperparameters

#Naive Bayes - HP Tuning
param_grid = [{'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}] 

# Multinomial Naive Bayes
mnb_uni_cv = Tune_Model(df, model_type = MultinomialNB(), param_grid = param_grid, ngram_range = (1,1), tf_idf = False)  
mnb_bi_cv = Tune_Model(df, model_type = MultinomialNB(), param_grid = param_grid, ngram_range = (2,2), tf_idf = False) 
mnb_combo_cv = Tune_Model(df, model_type = MultinomialNB(), param_grid = param_grid, ngram_range = (1,2), tf_idf = False) 
mnb_uni_tf = Tune_Model(df, model_type = MultinomialNB(), param_grid = param_grid, ngram_range = (1,1), tf_idf = True)  
mnb_bi_tf = Tune_Model(df, model_type = MultinomialNB(), param_grid = param_grid, ngram_range = (2,2), tf_idf = True)
mnb_combo_tf = Tune_Model(df, model_type = MultinomialNB(), param_grid = param_grid, ngram_range = (1,2), tf_idf = True)  

# Complement Naive Bayes
cnb_uni_cv = Tune_Model(df, model_type = ComplementNB(), param_grid = param_grid, ngram_range = (1,1), tf_idf = False)  
cnb_bi_cv = Tune_Model(df, model_type = ComplementNB(), param_grid = param_grid, ngram_range = (2,2), tf_idf = False) 
cnb_combo_cv = Tune_Model(df, model_type = ComplementNB(), param_grid = param_grid, ngram_range = (1,2), tf_idf = False) 
cnb_uni_tf = Tune_Model(df, model_type = ComplementNB(), param_grid = param_grid, ngram_range = (1,1), tf_idf = True)  
cnb_bi_tf = Tune_Model(df, model_type = ComplementNB(), param_grid = param_grid, ngram_range = (2,2), tf_idf = True)
cnb_combo_tf = Tune_Model(df, model_type = ComplementNB(), param_grid = param_grid, ngram_range = (1,2), tf_idf = True)  

# Linear SVC - HP Tuning
param_grid = [{'C' : [0.01, 0.1, 100, 1000] + list(range(1,20,1))}] # Tune C parameter
lsvc_uni_cv = Tune_Model(df, model_type = LinearSVC(), param_grid = param_grid, ngram_range = (1,1), tf_idf = False)  
lsvc_bi_cv = Tune_Model(df, model_type = LinearSVC(), param_grid = param_grid, ngram_range = (2,2), tf_idf = False) 
lsvc_combo_cv = Tune_Model(df, model_type = LinearSVC(), param_grid = param_grid, ngram_range = (1,2), tf_idf = False) 
lsvc_uni_tf = Tune_Model(df, model_type = LinearSVC(), param_grid = param_grid, ngram_range = (1,1), tf_idf = True)  
lsvc_bi_tf = Tune_Model(df, model_type = LinearSVC(), param_grid = param_grid, ngram_range = (2,2), tf_idf = True)
lsvc_combo_tf = Tune_Model(df, model_type = LinearSVC(), param_grid = param_grid, ngram_range = (1,2), tf_idf = True)  

# k-NN Parameters 
param_grid = [{'n_neighbors' : list(range(1,50)),
                'weights' : ['uniform','distance'],
                'metric' : ['minkowski','euclidean','manhattan']}] 

# k-NN - HP Tuning
knn_uni_cv = Tune_Model(df, model_type = KNeighborsClassifier(), param_grid = param_grid, ngram_range = (1,1), tf_idf = False) 
knn_bi_cv = Tune_Model(df, model_type = KNeighborsClassifier(), param_grid = param_grid, ngram_range = (2,2), tf_idf = False) 
knn_combo_cv = Tune_Model(df, model_type = KNeighborsClassifier(), param_grid = param_grid, ngram_range = (1,2), tf_idf = False) 
knn_uni_tf = Tune_Model(df, model_type = KNeighborsClassifier(), param_grid = param_grid, ngram_range = (1,1), tf_idf = True) 
knn_bi_tf = Tune_Model(df, model_type = KNeighborsClassifier(), param_grid = param_grid, ngram_range = (2,2), tf_idf = True) 
knn_combo_tf = Tune_Model(df, model_type = KNeighborsClassifier(), param_grid = param_grid, ngram_range = (1,2), tf_idf = True) 

#%% Print Hyperparameter Tuning Results 

def top_hp(data, name, model = 'knn') :
    
    """ Extract top performing hyperparamter from ML validation sets """
    
    if model == 'knn':
        
        col_save = ['model', 'param_n_neighbors', 'param_weights', 'mean_test_score', 'std_test_score']
        best_p = data[data['rank_test_score'] == 1]
        best_p['model'] = name
        best_p = best_p[col_save]
        
    if model == 'nb':
        
        col_save = ['model', 'param_alpha','mean_test_score', 'std_test_score']
        best_p = data[data['rank_test_score'] == 1]
        best_p['model'] = name
        best_p = best_p[col_save]
    
    if model == 'svm':
        
        col_save = ['model', 'param_C', 'mean_test_score', 'std_test_score']
        best_p = data[data['rank_test_score'] == 1]
        best_p['model'] = name
        best_p = best_p[col_save]
        
    return(best_p)

cv1 = top_hp(lsvc_uni_cv, name = "lsvc_uni_cv", model = "svm")
cv2 = top_hp(lsvc_bi_cv, name = "lsvc_bi_cv", model = "svm")
cv3 = top_hp(lsvc_combo_cv, name = "lsvc_combo_cv", model = "svm")
cv4 = top_hp(lsvc_uni_tf, name = "lsvc_uni_tf", model = "svm")
cv5 = top_hp(lsvc_bi_tf, name = "lsvc_bi_tf", model = "svm")
cv6 = top_hp(lsvc_combo_tf, name = "lsvc_combo_tf", model = "svm")

cv7 = top_hp(knn_uni_cv, name = "knn_uni_cv", model = 'knn')
cv8 = top_hp(knn_bi_cv, name = "knn_bi_cv", model = 'knn')
cv9 = top_hp(knn_combo_cv, name = "knn_combo_cv", model = 'knn')
cv10 = top_hp(knn_uni_tf, name = "knn_uni_tf", model = 'knn')
cv11 = top_hp(knn_bi_tf, name = "knn_bi_tf", model = 'knn')
cv12 = top_hp(knn_combo_tf, name = "knn_combo_tf", model = 'knn')

cv13 = top_hp(mnb_uni_cv, name = "mnb_uni_cv", model = 'nb')
cv14 = top_hp(mnb_bi_cv, name = "mnb_bi_cv", model = 'nb')
cv15 = top_hp(mnb_combo_cv, name = "mnb_combo_cv", model = 'nb')
cv16 = top_hp(mnb_uni_tf, name = "mnb_uni_tf", model = 'nb')
cv17 = top_hp(mnb_bi_tf, name = "mnb_bi_tf", model = 'nb')
cv18 = top_hp(mnb_combo_tf, name = "mnb_combo_tf", model = 'nb')

cv19 = top_hp(cnb_uni_cv, name = "cnb_uni_cv", model = 'nb')
cv20 = top_hp(cnb_bi_cv, name = "cnb_bi_cv", model = 'nb')
cv21 = top_hp(cnb_combo_cv, name = "cnb_combo_cv", model = 'nb')
cv22 = top_hp(cnb_uni_tf, name = "cnb_uni_tf", model = 'nb')
cv23 = top_hp(cnb_bi_tf, name = "cnb_bi_tf", model = 'nb')
cv24 = top_hp(cnb_combo_tf, name = "cnb_combo_tf", model = 'nb')

cv_df_svm = pd.concat([cv1, cv2, cv3, cv4, cv5, cv6], ignore_index=True)
cv_df_knn = pd.concat([cv7, cv8, cv9, cv10, cv11, cv12], ignore_index=True)
cv_df_nb = pd.concat([cv13, cv14, cv15, cv16, cv17, cv18, 
                      cv19, cv20, cv21, cv22, cv23, cv24], ignore_index=True)

del cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10, cv11, cv12
del cv13, cv14, cv15, cv16, cv17, cv18, cv19, cv20, cv21, cv22, cv23, cv24
print("\n")
print("=================== SVM HYPERPARAMTER RESULTS ===================")
print(cv_df_svm)
print("\n")
print("=================== k-NN HYPERPARAMTER RESULTS ===================")
print(cv_df_knn)
print("\n")
print("=================== NB HYPERPARAMTER RESULTS ===================")
print(cv_df_nb)

#%% Save Cross Validation Results and Export to Pickle File

#Save CV Results to one dicitonary  
m1_tune =  {"mnb_uni_cv" : mnb_uni_cv, "mnb_bi_cv" : mnb_bi_cv, "mnb_combo_cv" : mnb_combo_cv, 
            "mnb_uni_tf" : mnb_uni_tf, "mnb_bi_tf" : mnb_bi_tf, "mnb_combo_tf" : mnb_combo_tf,
            "cnb_uni_cv" : cnb_uni_cv, "cnb_bi_cv" : cnb_bi_cv, "cnb_combo_cv" : cnb_combo_cv,
            "cnb_uni_tf" : cnb_uni_tf, "cnb_bi_tf" : cnb_bi_tf, "cnb_combo_tf" : cnb_combo_tf,
            "knn_uni_cv" : knn_uni_cv, "knn_bi_cv" : knn_bi_cv, "knn_combo_cv" : knn_combo_cv,
            "knn_uni_tf" : knn_uni_tf, "knn_bi_tf" : knn_bi_tf, "knn_combo_tf" : knn_combo_tf,
            "lsvc_uni_cv" : lsvc_uni_cv, "lsvc_bi_cv" : lsvc_bi_cv, "lsvc_combo_cv" : lsvc_combo_cv,
            "lsvc_uni_tf" : lsvc_uni_tf, "lsvc_bi_tf" : lsvc_bi_tf, "lsvc_combo_tf" : lsvc_combo_tf}

# Export to Pickle File 
# file_to_write = open("m1_tune.pkl", "wb")
# pickle.dump(m1_tune, file_to_write) 

#%% Load Pickle File - CV Results

# import pickle
# import pandas as pd

# pickle_in = open("m1_tune.pkl","rb")
# df = pickle.load(pickle_in)
# del(pickle_in)






