# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:22:58 2018

@author: anjalidharmik
"""

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
stop = stopwords.words('english')

#from textblob import Word

def cleaning_dataset(input_file):
    df_train = pd.read_csv(input_file)
    
    ##Analyze the training dataset...
    #print('Total number of question pairs for training: {}'.format(len(df_train)))
    #
    #print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
    #
    ##combined question1 and question2 in a single list as a series...
    #qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
    ##identify unique values in the questions list...
    ##option -> len(set(qids))
    #print('Total number of questions in the training data: {}'.format(len(np.unique(qids))))
    #
    ##number of duplicate question and qids.value_counts() to identify a question occurences...
    #print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
    
    #Pre-Processing...
    #convert all questions in string format...
    df_train['question1'] = df_train['question1'].astype(str)
    df_train['question2'] = df_train['question2'].astype(str)
    
    #convert all questions in lower case...
    df_train['question1'] = df_train['question1'].apply(lambda question1: question1.lower())
    df_train['question2'] = df_train['question2'].apply(lambda question2: question2.lower())
    
    #Remove of Stop Words from questions...
    df_train['question1'] = df_train['question1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df_train['question2'] = df_train['question2'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    #df_train['question1'] = df_train['question1'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    #df_train['question2'] = df_train['question2'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    #remove special characters from questions...
    df_train['question1'] = df_train['question1'].str.replace('\W', ' ')
    df_train['question2'] = df_train['question2'].str.replace('\W', ' ')
    
    return df_train


