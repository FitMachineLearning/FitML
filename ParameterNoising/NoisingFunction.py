import numpy as np
import keras
import gym

import pygal
import os
import h5py
import matplotlib.pyplot as plt
import math

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


#nitialize the Reward predictor model
Qmodel = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(32, activation='relu', input_dim=1))
Qmodel.add(Dropout(0.5))
#Qmodel.add(Dense(256, activation='relu'))
#Qmodel.add(Dropout(0.5))
#Qmodel.add(Dense(256, activation='tanh'))
#Qmodel.add(Dropout(0.5))
#Qmodel.add(Dense(256, activation='relu'))
#Qmodel.add(Dropout(0.5))
#Qmodel.add(Dense(512, activation='relu'))
#Qmodel.add(Dropout(0.2))
#Qmodel.add(Dense(256, activation='relu'))
#Qmodel.add(Dropout(0.2))

Qmodel.add(Dense(1))
#opt = optimizers.adam(lr=learning_rate)
opt = optimizers.RMSprop()
Qmodel.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


def add_noise(mu):
    sig = 0.15 #Sigma = width of the standard deviaion
    #mu = means
    x =   np.random.rand(1) #probability of doing x
    return mu + np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

add_noise = np.vectorize(add_noise,otypes=[np.float])

for layer in Qmodel.layers:
    #g=layer.get_config()
    h=layer.get_weights()
    #print (g)
    if np.alen(h)>0:

        print ("Layer ==>",h[0])

        print ("Layer ==> with noise",add_noise(h[0]))

print("end")
