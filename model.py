# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:02:53 2018

@author: anjalidharmik
"""

from keras.models import load_model

import pandas as pd

import Siamene_LSTM_network
import pre_processing

from operator import itemgetter

#initialized required parameters for LSTM network...
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 10
VALIDATION_SPLIT = 0.1
RATE_DROP_LSTM = 0.17
RATE_DROP_DENSE = 0.25
NUMBER_LSTM = 50
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'

def train_dataset_model(df):
    df['question1'] = df['question1'].astype(str)
    df['question2'] = df['question2'].astype(str)
    
    question1 = df['question1'].tolist()
    question2 = df['question2'].tolist()
    is_duplicate = df['is_duplicate'].tolist()
    
    ## creating questions pairs
    questions_pair = [(x1, x2) for x1, x2 in zip(question1, question2)]
    print("----------created questions pairs-----------")
    
    # creating word embedding meta data for word embedding 
    tokenizer, embedding_matrix = pre_processing.word_embed_meta_data(question1 + question2,  EMBEDDING_DIM)
    embedding_meta_data = {'tokenizer': tokenizer,'embedding_matrix': embedding_matrix}
    print("----------created word embedding meta data-----------")
    
    #SiameneBiLSTM is a class for  Long short Term Memory networks
    siamese = Siamene_LSTM_network.SiameneLSTM(EMBEDDING_DIM ,MAX_SEQUENCE_LENGTH, NUMBER_LSTM, NUMBER_DENSE_UNITS, RATE_DROP_LSTM, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT)
    model_path = siamese.train_model(questions_pair, is_duplicate, embedding_meta_data, model_save_directory='./')
    
    #load the train data in model...
    model = load_model(model_path)
    print("----------model trained-----------")
    return model

def test_dataset_model(df_test,model):
    
    df_test['question1'] = df_test['question1'].astype(str)
    df_test['question2'] = df_test['question2'].astype(str)
    
    question1_test = df_test['question1'].tolist()
    question2_test = df_test['question2'].tolist()
    
    ## creating questions pairs
    questions_test_pair = [(x1, x2) for x1, x2 in zip(question1_test, question2_test)]
    print("----------created test dataset-----------")
    
    
    test_data_x1, test_data_x2, leaks_test = pre_processing.create_test_data(tokenizer,questions_test_pair, MAX_SEQUENCE_LENGTH)
    
    #predict the results
    preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
    print("----------predicted test results-----------")
    
    #mapping results with input test data...
    results = [(x, y, z) for (x, y), z in zip(questions_test_pair, preds)]
    print("----------mapping test results-----------")
    
    results.sort(key=itemgetter(2), reverse=True)
    
    return results