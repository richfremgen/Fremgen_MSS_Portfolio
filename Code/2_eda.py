#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:06:52 2023

@author: richardfremgen
"""

import pandas as pd
from matplotlib import pyplot as plt
from plotnine import *
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import warnings

warnings.filterwarnings('ignore')

os.chdir('/Users/richardfremgen/Documents/Portfolio/Code/Data')

df = pd.read_pickle("./df_clean_75.pkl")  

#%% Bar Chart of Document Count by Sentiment Type

df['sentiment'] = df['sentiment'].str.title()
(ggplot(df, aes('sentiment', fill='sentiment'))
 + geom_bar()
 + geom_text(aes(label=after_stat('prop*100'), group = 1),
     stat = 'count',
     va = 'center',
     format_string='{:.0f}%',
     size  = 16,
     position = position_stack(vjust = 0.5)) 
 + ggtitle('Document Count by Sentiment')
 + labs(x = "", y = "", fill = "")
 + scale_fill_manual(values= ['red', 'orange', 'green'], guide = False)
 + theme_bw()
 + theme(
     plot_title = element_text(size=16, family="Arial", face="bold"), 
     text = element_text(colour="black", family="Arial", face="bold", size=16),
     axis_text_x = element_text(colour="black", family="Arial", face="bold", size=14),
     axis_text_y = element_text(colour="black", family="Arial", face="bold", size=14))
) 

#%% Word Clouds

df_neg = df[df['sentiment'] == "Negative"]
df_pos = df[df['sentiment'] == "Positive"]

# Negative Documents
text = " ".join(cat.split()[1] for cat in df_neg.sentence)

# Creating word_cloud with text as argument in .generate() method
word_cloud = WordCloud(width = 800, height = 800, min_font_size = 10,
                       collocations = False, 
                       background_color = 'white',
                       colormap= "Reds").generate(text) 

# Display the generated Word Cloud
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show() 

# Positive Documents
text = " ".join(cat.split()[1] for cat in df_pos.sentence)
# Creating word_cloud with text as argument in .generate() method
word_cloud = WordCloud(width = 800, height = 800, min_font_size = 10, 
                       collocations = False, 
                       background_color = 'white',
                       colormap= "Greens").generate(text) 

# Display the generated Word Cloud
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()






