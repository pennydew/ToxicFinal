import pickle
import time

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from preprocess import clean_text

pd.set_option('display.max_columns', 500)

data_train = pd.read_csv("resources/train.csv", low_memory=False)
data_test = pd.read_csv("resources/test.csv", low_memory=False)

df = pd.concat([data_train, data_test], sort=False)
df = df.reset_index(drop=True)
df.comment_text = df.comment_text.map(lambda x: clean_text(x))

start = time.time()
comments = df.comment_text
pickle.dump(comments, open("test.pkl", "wb"))
corpus_comments = pickle.load(open("test.pkl", "rb"))
end = time.time()

# number of seconds taken
step = end - start

print(step)

train_cl = df[: data_train.shape[0]]
test_cl = df[data_train.shape[0]:]

# six classes
class_list = list(data_train)[2:]

trainY = train_cl[class_list].values

train_comments = train_cl.comment_text
test_comments = test_cl.comment_text

n_words = 100000
tokenizer = Tokenizer(num_words=n_words)  # todo: need to think more on num_words
tokenizer.fit_on_texts(list(train_comments) + list(test_comments))

train_tokenized = tokenizer.texts_to_sequences(train_comments)
test_tokenized = tokenizer.texts_to_sequences(test_comments)

all_words = [len(comment) for comment in train_comments]
print("avg length: ", np.mean(all_words))
print("max length: ", np.max(all_words))
print("std: ", np.std(all_words))

n_unique = len(tokenizer.word_index.items())
print("number of unique words: ", n_unique)

trainX = pad_sequences(train_tokenized, maxlen=500, padding='post')
testX = pad_sequences(test_tokenized, maxlen=500, padding='post')

# get word embeddings
embeddings_index = dict()
f = open('resources/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print("{0} word vectors".format(len(embeddings_index)))

# define weight matrix
embedding_matrix = np.zeros((n_words, 100))
for word, index in tokenizer.word_index.items():
    if index > n_words:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# develop lstm model
model = Sequential()
glove_dim = 100
model.add(Embedding(n_words, glove_dim, 50, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optiizer='adam', metrics=['accuracy'])


