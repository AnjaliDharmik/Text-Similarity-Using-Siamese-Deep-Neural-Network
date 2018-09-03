# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:17:46 2018

@author: anjalidharmik
"""

import NLP
import model

#train model...
input_train_dataset  = "/home/anjalidharmik/aatasks/input/train.csv"

df = NLP.cleaning_dataset(input_train_dataset)

train_model = model.train_dataset_model(df)

#test model...
input_test_dataset  = "/home/anjalidharmik/aatasks/input/test.csv"

df_test = NLP.cleaning_dataset(input_test_dataset)
test_results = model.test_dataset_model(df_test,train_model)
test_results.to_csv("/home/anjalidharmik/aatasks/output/results.csv")
