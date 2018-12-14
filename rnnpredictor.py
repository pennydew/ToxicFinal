#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:37:11 2018

@author: wenyijones
"""
import csv
import tensorflow as tf
import tflearn
import pickle
import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

test = []
with open('resources/test.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        test.append(row)
        
train = []
with open('resources/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        train.append(row)
train = train[1:1001]
        
        
data = pickle.load(open("resources/tflearn_model/dnn/training_data",'rb'))    
vocabulary = data['vocabulary']
train_x = data['train_x']
train_y = data['train_y']


# RNN
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])], name='input')
net = tflearn.embedding(net, input_dim=10000, output_dim=6)
net = tflearn.lstm(net, 6)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(net, tensorboard_dir='resources/log')
model.load('resources/tflearn_model/dnn/tfmodel.tflearn')


def process(sentence):
    stemmer = LancasterStemmer()
    ignore = ['.', '!', ',', '?', '``', "''"]
    tokens = nltk.word_tokenize(sentence.lower().decode('utf-8'))
    sentence_max_len = 200
    bag = [0] * 200
    for i in range(0, min(sentence_max_len, len(tokens))):
        w = tokens[i][0]
        if w not in ignore and w.isalpha():
            w = ''.join(e for e in w if e.isalnum())
            w = stemmer.stem(w.lower())
            try:
                bag[i] = vocabulary.index(tokens[i])
            except:
                bag[i] = 0
    return (np.array(bag))
    
def predict(moodel, sentence):
    result = model.predict([process(sentence)])[0]
    return result

test_prediction = []
for row in test:
    sentence = row[1]
    test_prediction.append(predict(model, sentence))

res = [0] * 6
for i in range(0, len(test_prediction)):
    for j in range(1,7):
        # toxic
        if (test_label[i][j] == '-1'):
            res[j - 1] = res[j - 1] + 1
        else:
            if test_prediction[i][j - 1] > 0.5:
                test_res = '1'
            else:
                test_res = '0'
            if test_res == test_label[i][j]:
                res[j - 1] = res[j - 1] + 1
precision = [float(w) / len(test_prediction) * 100 for w in res]
print('test precision: {}'.format(precision))

