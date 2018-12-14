#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:57:19 2018

@author: wenyijones
"""

import csv
import tensorflow as tf
import tflearn
import numpy as np
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer

data = []
with open('resources/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data.append(row)
data = data[1:1001]
def process_data(data):
    vocabulary = []
    documents = []
    ignore = {'.', '!', ',', '?', '``', "''"}
    stemmer = LancasterStemmer()
    sentence_max_len = 200
    sen = ""
    for row in data:
        word = nltk.word_tokenize(row[1].lower().decode('utf-8'))
        if len(word) < sentence_max_len:
            vocabulary.extend(word)
            documents.append((word, row[2:]))
            
    processed_vocabulary = []
    for i in range(0, len(vocabulary)):
        w = vocabulary[i]
        if w not in ignore and w.isalpha():
            w = ''.join(e for e in w if e.isalnum())
            processed_vocabulary.append(stemmer.stem(w.lower()))
            
    processed_vocabulary = set(processed_vocabulary)
    processed_vocabulary = list(processed_vocabulary)
    return processed_vocabulary, documents, sentence_max_len, sen

vocabulary, documents, sentence_max_len, sen = process_data(data)


def generate_training_data(vocabulary, documents, sentence_max_len):
    training = []
    stemmer = LancasterStemmer()
    ignore = ['.', '!', ',', '?', '``', "''"]
    for doc in documents:
        bag = [0] * 200
        tokens = doc[0]
        for i in range(0, min(sentence_max_len, len(tokens))):
            w = tokens[i][0]
            if w not in ignore and w.isalpha():
                w = ''.join(e for e in w if e.isalnum())
                tokens[i] = stemmer.stem(w.lower())
                try:
                    bag[i] = vocabulary.index(tokens[i])
                except:
                    bag[i] = 0

        target_row = list(doc[1])
        training.append([bag, target_row])
        
    training = np.array(training)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x, train_y

train_x, train_y = generate_training_data(vocabulary, documents, sentence_max_len)   


# rnn 
def train(train_x, train_y, vocabulary):
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(train_x[0])], name='input')
    net = tflearn.embedding(net, input_dim=10000, output_dim=6)
    net = tflearn.lstm(net, 6)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(net, tensorboard_dir='resources/log')
    model.fit(train_x, train_y, n_epoch=20, batch_size=8, show_metric=True)
    model.save('resources/tflearn_model/dnn/tfmodel.tflearn')
    pickle.dump({'vocabulary':vocabulary,
                 'train_x': train_x,
                 'train_y': train_y}, open("resources/tflearn_model/dnn/training_data",'wb'))

train(train_x, train_y, vocabulary)
    
    

        