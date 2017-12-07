'''
Bipedal Walker solution by Michel Aka
https://github.com/FitMachineLearning/FitML/
Policy Gradient
'''
import numpy as np
import keras
import gym
import os
import h5py
import math

import matplotlib
import scipy

from matplotlib import pyplot as plt
#from scipy import misc
from keras.models import Sequential
from keras.layers import Conv2D


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers

img_dim = 160
num_env_variables = 24 # each sate is 2 frames of the pong game
num_env_actions = 4
num_initial_observation = 1
randomGameEvery = 20
learning_rate =  0.008
apLearning_rate = 0.003
weigths_filename = "Walker-PG-v2-weights.h5"
apWeights_filename = "Walker-PG-v2-weights-ap.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.97
max_memory_len = 15000
starting_explore_prob = 0.6
training_epochs = 6
train_every_num_games =10
mini_batch = 256
load_previous_weights = False
observe_and_train = True
save_weights = True
num_games_to_play = 3000


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('BipedalWalker-v2')
env.reset()

#initialize training matrix with random states and actions
dataX = np.random.random(( 5,num_env_variables+num_env_actions ))
#Only one output for the total score / reward
dataY = np.random.random((5,1))

#initialize training matrix with random states and actions
apdataX = np.random.random(( 5,num_env_variables ))
apdataY = np.random.random((5,num_env_actions))

def custom_error(y_true, y_pred, Qsa):
    cce=0.001*(y_true - y_pred)*Qsa
    return cce


#nitialize the Reward predictor model
model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(2048, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dropout(0.25))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(dataY.shape[1]))

opt = optimizers.adam(lr=learning_rate)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(2048, activation='tanh', input_dim=apdataX.shape[1]))
action_predictor_model.add(Dropout(0.1))
#action_predictor_model.add(Dense(128, activation='relu'))
#action_predictor_model.add(Dropout(0.1))
action_predictor_model.add(Dense(apdataY.shape[1], activation='tanh'))

opt2 = optimizers.adam(lr=apLearning_rate)

action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])



#load previous model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+weigths_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        model.load_weights(weigths_filename)
    else:
        print("File ",weigths_filename," does not exis. Retraining... ")

#load previous action predictor model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+ apWeights_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        action_predictor_model.load_weights(apWeights_filename)
    else:
        print("File ",apWeights_filename," does not exis. Retraining... ")





#Record first 500 in a sequence and add them to the training sequence
total_steps = 0

memorySA = []
memoryS = []
memoryA = []
memoryR = []
memoryAdv = []

mstats = []
num_games_won = 0


#takes a single game frame as input
#preprocesses before feeding into model
def preprocessing(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I = scipy.misc.imresize(I,size=(img_dim,img_dim))
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  #I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  I = I/255
  return I.astype(np.float).ravel() #flattens

def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = model.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward


def GetRememberedOptimalPolicy(qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy

def SmartCrossEntropy(current_optimal_policy):
    sce = np.zeros(shape=(num_env_actions))
    #print("current_optimal_policy", current_optimal_policy)
    for i in range(num_env_actions):
        sce[i] = current_optimal_policy[i] + sce_range * (np.random.rand(1)*2 - 1)
        if sce[i] > 1:
            sce[i] = 1.0
        if sce[i] < -1:
            sce[i] = -1
    #print("current_optimal_policy", current_optimal_policy)
    #print("sce", sce)
    return sce


if observe_and_train:

    #Play the game 500 times
    for game in range(num_games_to_play):
        gameSA = []
        gameS = []
        gameA = []
        gameR = []
        gameAdv = [] #advantage
        num_points = 0

        previous_state = np.zeros(img_dim*img_dim)
        #Get the Q state
        qs = env.reset()


        #print("qs ", qs)
        for step in range (7000):

            #qs = preprocessing(qs)
            #if np.array_equal(qs,previous_state):
                #print("Previous state = to Qstate we have a problem")
            sequenceQS = qs
            #sequenceQS = np.abs(qs - previous_state)

            #if step%10==0:
                #plt.imshow(np.reshape(sequenceQS,(-1,img_dim)))
                #plt.show()

            if game < num_initial_observation or game%randomGameEvery==0:
                #take a radmon action
                #a = keras.utils.to_categorical(env.action_space.sample(),num_env_actions)[0]
                a = env.action_space.sample()
                #if 'info' in locals():
                #    print(step, "random action ",a,"reward",r, "info", info)
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob:
                    #take a random action
                    a = env.action_space.sample()

                    #a = keras.utils.to_categorical(env.action_space.sample(),num_env_actions)[0]

                else:


                    predX = np.zeros(shape=(1,num_env_variables))
                    predX[0] = sequenceQS

                    #print("trying to predict reward at qs_a", predX[0])
                    a = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))
                    a = a[0]
                    #if step %100 == 0:
                        #print("Prefered action ", a)
                    #print("predicted action",a)
                    #a = np.argmax(predictedRewards)

                    #a = keras.utils.to_categorical(a,num_env_actions)[0]



            env.render()
            qs_a = np.concatenate((sequenceQS,a), axis=0)


            #get the target state and reward
            s,r,done,info = env.step(a)
            #record only the first x number of states


            '''
            if step ==0:
                gameSA[0] = qs_a
                #gameS[0] = sequenceQS
                gameR[0] = np.array([r])
                #gameA[0] = np.array([a])
            else:
            '''
            gameSA.append(qs_a)
            gameS.append(sequenceQS)
            gameR.append([r])
            gameA.append(a)
            gameAdv.append([0])

            if step > 1500:
                done = True

            if r >=1:
                num_games_won +=1
                num_points +=1
            if done:
                #done = True
                if num_points >=20:
                    print("GAME WON ***")
                meanR = 0
                if len(memoryR) > 0:
                    #Policy Gradient Update
                    _memR = np.asarray(memoryR)
                    meanR = _memR.mean()


                tX = np.zeros(shape=(1,num_env_variables))
                tY = np.zeros(shape=(1,num_env_actions))
                tA = np.zeros(shape=(1,))
                for i in range(0,len(gameR)):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameR[(gameR.shape[0]-1)-i][0])
                        gameR[(len(gameR)-1)-i][0] = gameR[(len(gameR)-1)-i][0]
                    else:
                        #print("local error before Bellman", gameR[(gameR.shape[0]-1)-i][0],"Next error ", gameR[(gameR.shape[0]-1)-i+1][0])
                        gameR[(len(gameR)-1)-i][0] = gameR[(len(gameR)-1)-i][0]+b_discount*gameR[(len(gameR)-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameR[(gameR.shape[0]-1)-i][0])
                    if len(memoryR) >0:
                        #expectedReward = predictTotalRewards(gameS[i],gameA[i])

                        adv = gameR[(len(gameR)-1)-i][0] - meanR

                        if(adv <0):
                            adv = -0.01 / adv
                        else:
                            adv = adv + 1.0
                        gameAdv[(len(gameR)-1)-i][0] = adv

                        training_repeats = int(math.tanh(adv)*4)
                        if i%100 ==0:
                            print("advantage",adv,"training_repeats", training_repeats)
                            #print("expectedReward", expectedReward, "advantage",adv,"training_repeats", training_repeats)
                        tX[0] = np.asarray(gameS[(len(gameR)-1)-i])
                        tY[0] = np.asarray(gameA[(len(gameR)-1)-i])
                        tA[0] = np.asarray(gameAdv[(len(gameR)-1)-i])


                        action_predictor_model.train_on_batch(tX.reshape(1,tX.shape[1]),tY.reshape(1,tY.shape[1]))

                    if i==len(gameR)-1:
                        print("Training Game #",game,"#scores", num_points, " total # scores ", num_games_won,"avg scores per match ",num_games_won/(game+1), "memory ",len(memoryR)," finished with headscore ", gameR[(len(gameR)-1)-i][0])


                if len(memoryR) > 0:
                    #tX = np.zeros(shape=(1,num_env_variables))
                    #tY = np.zeros(shape=(1,num_env_actions))
                    #print("training actor")
                    tX = np.asarray(gameS)
                    tY = np.asarray(gameA)
                    sw = np.asarray(gameAdv)
                    #action_predictor_model.fit(tX,tY,batch_size=128,sample_weight=sw[:,0],nb_epoch=1,verbose=0)
                    #action_predictor_model.fit(tX.reshape(1,tX.shape[1]),tY.reshape(1,tY.shape[1]),sample_weight=gameAdv,epochs=3,verbose=0)
                #Add experience to memory
                memorySA = memorySA+gameSA
                memoryS = memoryS+gameS
                memoryR = memoryR+gameR
                memoryA = memoryA+gameA
                memoryAdv = memoryAdv+gameAdv

                #tempGameA = tempGameA[1:]
                #tempGameS = tempGameS[1:]
                #tempGameRR = tempGameRR[1:]
                #tempGameR = tempGameR[1:]
                #tempGameSA = tempGameSA[1:]

                '''
                for i in range(gameR.shape[0]):
                    if np.random.rand(1) < 1:
                        tempGameSA = np.vstack((tempGameSA,gameSA[i]))
                        tempGameR = np.vstack((tempGameR,gameR[i]))
                    '''


                '''
                if memoryR.shape[0] ==1:
                    memoryR = tempGameR
                    memorySA = tempGameSA
                else:
                    memorySA = np.concatenate((memorySA,tempGameSA),axis=0)
                    memoryR = np.concatenate((memoryR,tempGameR),axis=0)
                '''


                #if memory is full remove first element
                if len(memoryR) >= max_memory_len:
                    memoryR = memoryR[np.alen(gameR):]
                    memorySA = memorySA[np.alen(gameR):]
                    memoryS = memoryS[np.alen(gameR):]
                    memoryA = memoryA[np.alen(gameR):]
                    memoryAdv = memoryAdv[np.alen(gameR):]


                    #print("memory full. mem len ", np.alen(memoryX))
                    #for l in range(np.alen(gameR)):
                        #memorySA = np.delete(memorySA, 0, axis=0)
                        #memoryR = np.delete(memoryR, 0, axis=0)
                        #memoryA = np.delete(memoryA, 0, axis=0)
                        #memoryS = np.delete(memoryS, 0, axis=0)


            #Update the states
            previous_state = np.copy(qs)
            qs=s


            #Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game%train_every_num_games == 0 and game>1:
                    #print("Training  game# ", game,"momory size", len(memorySA))

                    #training Reward predictor model
                    #model.fit(np.asarray(memorySA),np.asarray(memoryR), batch_size=mini_batch,nb_epoch=training_epochs,verbose=2)
                    tX = np.asarray(memoryS)
                    tY = np.asarray(memoryA)
                    sw = np.asarray(memoryAdv)
                    action_predictor_model.fit(tX,tY,batch_size=mini_batch,sample_weight=sw[:,0],nb_epoch=training_epochs,verbose=0)

                    #training action predictor model
                    #action_predictor_model.fit(memoryS,memoryA, batch_size=mini_batch, epochs=training_epochs,verbose=0)

            if done and game >= num_initial_observation:
                if save_weights and game%20 == 0:
                    #Save model
                    print("Saving weights")
                    model.save_weights(weigths_filename)
                    action_predictor_model.save_weights(apWeights_filename)

            if done:
                #Game ended - Break
                break


plt.plot(mstats)
plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
