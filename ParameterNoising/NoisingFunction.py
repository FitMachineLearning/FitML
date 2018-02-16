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
from keras import optimizers


#nitialize the Reward predictor model
Qmodel = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(32, activation='relu', input_dim=1))
Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(2, activation='relu'))
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

def add_noise_to_model(model_to_scramble):
    sz = len(model_to_scramble.layers)
    for k in range(sz):
        w = model_to_scramble.layers[k].get_weights()
        print("w ==>",w)
        if np.alen(w) >0:
            w[0] = add_noise(w[0])
            print("w / noise ==>",w)
        model_to_scramble.layers[k].set_weights(w )



print("end")
