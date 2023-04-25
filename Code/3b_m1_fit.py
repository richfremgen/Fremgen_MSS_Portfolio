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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import joblib
import time
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer, LancasterStemmer
import numpy as np
import nltk
import os
import time
import pickle

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
    
#%% Define Classic ML Model Function 

def ML_Model(data, model_name, ngram_range = (1,1), model_type = MultinomialNB(), tf_idf = False): 
    
    """ Run different configurations of Naive Bayes Models """
    
    data['sentiment'] = data['sentiment'].str.title()
    #data['sentiment'] = LabelEncoder().fit_transform(df['sentiment']) 
    
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
        
    # Split into 80-20 train-test sets
    X_train, X_test, y_train, y_test = train_test_split(text_counts, data['sentiment'], 
                                                        test_size=0.2, random_state=123) 
        
    sent_model = model_type.fit(X_train, y_train) 
    predictions = sent_model.predict(X_test) # Predict on test data 
    
    print(X_test.shape)
    # Get Classification Report
    # class_report = classification_report(y_test, predictions, digits=4)
    # print(class_report)
    
    # Confusion Matrix for Test Data
    # cm = confusion_matrix(y_test, predictions)
    # cm_display = ConfusionMatrixDisplay(cm, display_labels=model_type.classes_)
    # cm_display.plot(colorbar=False, cmap = "Blues")
    # csfont = {'fontname':'Arial'}
    # plt.xticks(**csfont, size = 10)
    # plt.yticks( **csfont, size = 10)
    # plt.xlabel('Predicted Sentiment', **csfont, size = 12, fontweight = "bold")
    # plt.ylabel('Actual Sentiment', **csfont, size = 12, fontweight = "bold")
    # plt.title(model_name, **csfont, size = 14, fontweight = "bold")
    # plt.show() 

    accuracy_score = round(metrics.accuracy_score(y_test,predictions),4)*100
    precision_score = round(metrics.precision_score(y_test, predictions, average= 'macro'), 4)
    recall_score = round(metrics.recall_score(y_test, predictions, average = 'macro'), 4)
    f1_score = round(metrics.f1_score(y_test, predictions, average= 'macro'), 4) 
    # auc_ovo = round(metrics.roc_auc_score(y_test, predictions, average='macro', multi_class='ovo'), 4)
    # auc_ovr = round(metrics.roc_auc_score(y_test, predictions, average='macro', multi_class='ovr'), 4)
           
    return_dict = {"model" : model_name, 
                    "acc" : accuracy_score, 
                    "precision" : precision_score,
                    "recall" : recall_score,
                    "f1_score" : f1_score}
                    # "auc_ovo" : auc_ovo,
                    # "auc_ovr" : auc_ovr} 
    
    return(pd.DataFrame([return_dict]))
 
#%% Get Results 

def get_results(data): 
    
    """ Run ML Models with tuned hyperparametrs and return test set metrics """
    
    # Linear SVC Models
    m1 =  ML_Model(data, model_name = "SVC CV U", ngram_range=(1,1), model_type = LinearSVC(C = 0.1), tf_idf = False)
    m2 =  ML_Model(data, model_name = "SVC CV B", ngram_range=(2,2), model_type = LinearSVC(C =1), tf_idf = False) 
    m3 =  ML_Model(data, model_name = "SVC CV UB", ngram_range=(1,2), model_type = LinearSVC(C =0.1), tf_idf = False) 
    m4 =  ML_Model(data, model_name = "SVC TFIDF U", ngram_range=(1,1), model_type = LinearSVC(C = 1.0), tf_idf = True)
    m5 =  ML_Model(data, model_name = "SVC TFIDF B", ngram_range=(2,2), model_type = LinearSVC(C = 16.0), tf_idf = True)
    m6 =  ML_Model(data, model_name = "SVC TFIDF UB", ngram_range=(1,2), model_type = LinearSVC(C =5.0), tf_idf = True) 
    
    # k-NN Models
    m7 = ML_Model(data, model_name = "k-NN CV U", ngram_range=(1,1), 
                  model_type = KNeighborsClassifier(metric =  'minkowski', n_neighbors = 1, weights = 'uniform'), tf_idf = False) 
    m8 = ML_Model(data, model_name = "k-NN CV B", ngram_range=(2,2), 
                  model_type = KNeighborsClassifier(metric =  'minkowski', n_neighbors = 1, weights = 'uniform'), tf_idf = False) 
    m9 = ML_Model(data, model_name = "k-NN CV UB", ngram_range=(1,2), 
                  model_type = KNeighborsClassifier(metric =  'minkowski', n_neighbors = 1, weights = 'uniform'), tf_idf = False) 
    m10 = ML_Model(data, model_name = "k-NN TFIDF U", ngram_range=(1,1), 
                   model_type = KNeighborsClassifier(metric =  'minkowski', n_neighbors = 29, weights = 'distance'), tf_idf = True) 
    m11 = ML_Model(data, model_name = "k-NN TFIDF B", ngram_range=(2,2), 
                   model_type = KNeighborsClassifier(metric =  'minkowski', n_neighbors = 11, weights = 'distance'), tf_idf = True) 
    m12 = ML_Model(data, model_name = "k-NN TFIDF UB", ngram_range=(1,2), 
                   model_type = KNeighborsClassifier(metric =  'minkowski', n_neighbors = 29, weights = 'distance'), tf_idf = True) 
    
    # Multinomial Naive Bayes
    m13 =  ML_Model(data, model_name = "MNB CV U", ngram_range=(1,1), model_type = MultinomialNB(alpha = 1), tf_idf = False)
    m14 =  ML_Model(data, model_name = "MNB CV B", ngram_range=(2,2), model_type = MultinomialNB(alpha = 10), tf_idf = False) 
    m15 =  ML_Model(data, model_name = "MNB CV UB", ngram_range=(1,2), model_type = MultinomialNB(alpha = 1), tf_idf = False) 
    m16 =  ML_Model(data, model_name = "MNB TFIDF U", ngram_range=(1,1), model_type = MultinomialNB(alpha = 0.1), tf_idf = True)
    m17 =  ML_Model(data, model_name = "MNB TFIDF B", ngram_range=(2,2), model_type = MultinomialNB(alpha = 0.00001), tf_idf = True)
    m18 =  ML_Model(data, model_name = "MNB TFIDF UB", ngram_range=(1,2), model_type = MultinomialNB(alpha = 0.1), tf_idf = True) 
    
    # Complement Naive Bayes
    m19 =  ML_Model(data, model_name = "CNB CV U", ngram_range=(1,1), model_type = ComplementNB(alpha = 1), tf_idf = False)
    m20 =  ML_Model(data, model_name = "CNB CV B", ngram_range=(2,2), model_type = ComplementNB(alpha = 0.00001), tf_idf = False) 
    m21 =  ML_Model(data, model_name = "CNB CV UB", ngram_range=(1,2), model_type = ComplementNB(alpha = 1), tf_idf = False) 
    m22 =  ML_Model(data, model_name = "CNB TFIDF U", ngram_range=(1,1), model_type = ComplementNB(alpha = 1), tf_idf = True)
    m23 =  ML_Model(data, model_name = "CNB TFIDF B", ngram_range=(2,2), model_type = ComplementNB(alpha = 0.00001), tf_idf = True)
    m24 =  ML_Model(data, model_name = "CNB TFIDF UB", ngram_range=(1,2), model_type = ComplementNB(alpha = 1), tf_idf = True) 
    
    m_df = pd.concat([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, 
                      m15, m16, m17, m18, m19, m20, m21, m22, m23, m24], ignore_index=True) 
    
    return(m_df) 

#%% Get Results from ML Models 

print("\n")
print("=================== TEST SET RESULTS ===================")
print(get_results(df))

#%% Final Model Function

def final_model(data): 
    
    """ Train and save final model  - Linaer SVC TF-IDF Combo """
    
    data['sentiment'] = data['sentiment'].str.title()
    
    X,y = data['sentence'].to_list(), data['sentiment'].to_list()
                
    # # Split into 80-20 train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123) 
        
    text_clf = Pipeline([('vect',  TfidfVectorizer(ngram_range = (1,2), binary=True)), 
                          ('clf',  LinearSVC(C =5.0))])
    
    #train model
    text_clf.fit(X_train, y_train) 
    
    # Print test accuracy
    # predictions = text_clf.predict(X_test)
    # accuracy_score = round(metrics.accuracy_score(y_test,predictions),4)*100
    # print(accuracy_score)
    
    return(text_clf)

#%%

# Save Final Model to pkl file for API Prediction
final_model = final_model(df)
joblib.dump(final_model, 'final_model.pkl') 

# Save Model to pickle file
# with open('final_model.pkl','wb') as f:
#     pickle.dump(final_model,f)




