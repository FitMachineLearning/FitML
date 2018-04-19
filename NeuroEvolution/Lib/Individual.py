import numpy as np
import keras
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout

class Individual:

    def __init__(self, generationID,indivID, network):
        self.generationID = generationID
        self.indivID = indivID
        self.network = network
        #self.mutationSigma = mutationSigma
        self.lifeScore = -10000

    def printme(self):
        return "Generation %2d Individual %2d network %s"%(self.generationID,+self.indivID,self.network)
        #print("say what?",self.network)

    def printNetwork(self):
        print("--- ID",self.indivID,"lifeScore ",self.lifeScore)
        sz = len(self.network.layers)
        #if largeNoise:
        #    print("Setting Large Noise!")
        for k in range(sz):
            w = self.network.layers[k].get_weights()
            if np.alen(w) >0 :
                print("k==>",k)
                print("w[0]",w[0])
                #print("w[1]",w[1])
                #print("w[3]",w[3])
