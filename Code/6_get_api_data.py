#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:44:52 2023

@author: richardfremgen
"""

# Load Packages
import os
import pickle
import pandas as pd
from newscatcherapi import NewsCatcherApiClient

# Newscatcher API Key
key = 'INSERT API KEY'
newscatcherapi = NewsCatcherApiClient(x_api_key=key)

#%%

constellation_articles = newscatcherapi.get_search_all_pages( q='\"Constellation Energy\"', 
                                                   from_='2023/02/04',
                                                   to_ = '2023/02/18',
                                                   lang = 'en',
                                                   countries='US', 
                                                   page_size=100, 
                                                   to_rank=1000,
                                                   sort_by='date',
                                                   page=1)

constellation_df = pd.DataFrame(constellation_articles['articles'])

duke_articles = newscatcherapi.get_search_all_pages( q='\"Duke Energy\"', 
                                                   from_='2023/02/04',
                                                   to_ = '2023/02/18',
                                                   lang = 'en',
                                                   countries='US', 
                                                   page_size=100, 
                                                   to_rank=1000,
                                                   sort_by='date',
                                                   page=1)

duke_df = pd.DataFrame(duke_articles['articles'])

dom_articles = newscatcherapi.get_search_all_pages( q='\"Dominion Energy\"', 
                                                   from_='2023/02/04',
                                                   to_ = '2023/02/18',
                                                   lang = 'en',
                                                   countries='US', 
                                                   page_size=100, 
                                                   to_rank=1000,
                                                   sort_by='date',
                                                   page=1)

dom_df = pd.DataFrame(dom_articles['articles'])

#%% Export Article to pickle file for analysis

# Set Directionary
os.chdir('/Users/richardfremgen/Documents/Portfolio/Code/Data')

# Combine dataframe into single dictionary
api_articles = {'constellation' : constellation_df, 
               'duke' : duke_df, 
               'dominion' : dom_df}

# Export to Pickle File 
# file_to_write = open("api_articles.pkl", "wb")
# pickle.dump(api_articles, file_to_write) 


