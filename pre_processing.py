# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:36:48 2018

@author: anjalidharmik
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

import numpy as np
import gc

def train_word2vec(documents, embedding_dim):
    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors
    
def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
#creating word vector with gensim model
def word_embed_meta_data(documents, embedding_dim):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(" ".join(documents))
    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix
 
#creating training dataset...   
def create_train_dev_set(tokenizer, questions_pair, is_similar, max_sequence_length, validation_split_ratio):
        
    questions1 = [x[0] for x in questions_pair]
    questions2 = [x[1] for x in questions_pair]
    train_questions_1 = tokenizer.texts_to_sequences(questions1)
    train_questions_2 = tokenizer.texts_to_sequences(questions2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_questions_1, train_questions_2)]
    
    train_padded_data_1 = pad_sequences(train_questions_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_questions_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)
    
    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]
    
    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))
    
    del train_padded_data_1
    del train_padded_data_2
    gc.collect()
    
    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]
    
    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

def create_test_data(tokenizer, test_questions_pair, max_sequence_length):
   
    test_questions1 = [x[0] for x in test_questions_pair]
    test_questions2 = [x[1] for x in test_questions_pair]

    test_questions_1 = tokenizer.texts_to_questions(test_questions1)
    test_questions_2 = tokenizer.texts_to_questions(test_questions2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_questions_1, test_questions_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_questions_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_questions_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test
    
