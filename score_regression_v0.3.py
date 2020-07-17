# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:14:00 2020

@author: barkov
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import re

def build_model():
  model = keras.Sequential([
    layers.Dense(16, activation='softmax', input_shape=[len(X[0])]),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(2)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.01,centered=True)
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
       

# load dataset
pbl = pd.read_table('complete_base.csv')
pbl.dropna(subset = ['UserReviews', 'CriticReviews', 'MetaScore', 'UserScore','Publisher', 'Developer'], inplace=True)
pbl = pbl.drop(pbl[pbl.CriticReviews < 5].index)
pbl = pbl.drop(pbl[pbl.UserReviews < 50].index)




list_space = []
for _ in range(len(pbl['Title'])):
    list_space.append(' ')
pbl['space'] = list_space
pbl['alltext'] = pbl.Title +pbl.space+ pbl.Platform +pbl.space+ pbl.Publisher +pbl.space+ pbl.Developer +pbl.space#+ pbl.Genres
#clean dataset
pbl['alltext'] = pbl['alltext'].str.lower()
#pass
#texts = pbl['alltext']
#texts = texts.tolist()
#shortword = re.compile(r'\W*\b\w{1,3}\b')
#for i in range(len(texts)):
#    texts[i] = shortword.sub('',texts[i])
#pbl['alltext'] = texts

vectorizer = CountVectorizer()
vectorizer.fit_transform(pbl['alltext'])
title_voc = vectorizer.vocabulary_
vector_title = vectorizer.transform(pbl['alltext'])
vector_title = vector_title.toarray()


dataset = pbl.values
# split into input (X) and output (Y) variables
#X = dataset[:,[4,6,8,]]

X = vector_title
Y = dataset[:,[5,7]]#dataset[:,[5,7]]
Y = Y.astype('float')
X = np.asarray(X).astype(np.float32)
Y = np.asarray(Y).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)

# evaluate model
model = build_model()
model.summary()

# delete large useless odjects
del list_space, X, Y, vector_title, dataset


# Параметр patience определяет количество эпох, проверяемых на улучшение
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

EPOCHS = 1000

history = model.fit(X_train, y_train, epochs=EPOCHS,validation_split = 0.2, verbose=1, callbacks=[early_stop])
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,100])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,500])
  plt.legend()
  plt.show()

plot_history(history)

loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(X_test).flatten()
   
pred1 = test_predictions[0::2]
pred2 = test_predictions[1::2]
plt.scatter(y_test[:,0], pred1)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.show()
plt.scatter(y_test[:,1], pred2)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.show()
#plt.axis('equal')
#plt.axis('square')
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])


print('Enter title, platform, publisher, developer, genre:')
test = []
#title = input()
title = 'God of War PS4 Sony Sony'
title = title.lower()
title = [title]
vectorizer.vocabulary_ = title_voc
vector = vectorizer.transform(title)
vector = vector.toarray()
test = np.asarray(vector).astype(np.float32)
test_predictions = model.predict(test).flatten()
print('Prediction is {} '.format(test_predictions))


