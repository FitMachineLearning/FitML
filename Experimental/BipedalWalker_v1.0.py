'''
BipedalWalker solution by Michel Aka
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured

using Selective Memory.
Uses Replay memory.
Starts to leap a little bit after 8 hours of training.
Uses different walk/run strategies after 8 hours.

'''
import numpy as np
import keras
import gym
import os
import h5py
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


num_env_variables = 24
num_env_actions = 4
num_initial_observation = 60
learning_rate =  0.005
apLearning_rate = 0.001
weigths_filename = "Waker-SM-v6-weights.h5"
apWeights_filename = "Walker-SM-v6-weights-ap.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.98
max_memory_len = 1000000
experience_replay_size = 10000
random_every_n = 70
starting_explore_prob = 0.15
training_epochs = 4
mini_batch = 256
load_previous_weights = False
observe_and_train = True
save_weights = True
save_and_load_memory_arrays = False
num_games_to_play = 6000


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
Qmodel = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(2048, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dropout(0.2))
#Qmodel.add(Dense(64, activation='tanh'))
#Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(dataY.shape[1]))
opt = optimizers.adam(lr=learning_rate)
Qmodel.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(2048, activation='tanh', input_dim=apdataX.shape[1]))
action_predictor_model.add(Dropout(0.2))
#action_predictor_model.add(Dense(32, activation='tanh'))
#action_predictor_model.add(Dropout(0.2))
action_predictor_model.add(Dense(apdataY.shape[1]))

opt2 = optimizers.adam(lr=apLearning_rate)
action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])



#load previous model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+weigths_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        Qmodel.load_weights(weigths_filename)
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


memorySA = np.zeros(shape=(1,num_env_variables+num_env_actions))
memoryS = np.zeros(shape=(1,num_env_variables))
memoryA = np.zeros(shape=(1,1))
memoryR = np.zeros(shape=(1,1))
memoryRR = np.zeros(shape=(1,1))

if save_and_load_memory_arrays:
    memorySA = np.load('memorySA.npy')
    memoryRR = np.save('memoryRR.npy')
    memoryS = np.save('memoryS.npy')
    memoryA = np.save('memoryA.npy')
    memoryR = np.save('memoryR.npy')


mstats = []

def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = Qmodel.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward


def GetRememberedOptimalPolicy(qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy

def addToMemory(reward,averegeReward,memMax):
    #diff = reward - ((averegeReward+memMax)/2)
    diff = reward - averegeReward
    prob = 0.05

    if reward > averegeReward:
        #print("add reward",reward,"average", averegeReward,"max",memMax)
        prob = prob + 0.95 * (diff / (5+math.fabs(diff)))
    else:
        prob = prob + 0.05/100 * (diff / (40+math.fabs(diff)))

    if np.random.rand(1)<=prob :
        #print("Adding reward",reward," based on prob ", prob)
        return True
    else:
        return False


if observe_and_train:

    #Play the game 500 times
    for game in range(num_games_to_play):
        gameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
        gameS = np.zeros(shape=(1,num_env_variables))
        gameA = np.zeros(shape=(1,num_env_actions))
        gameR = np.zeros(shape=(1,1))
        #Get the Q state
        qs = env.reset()
        #print("qs ", qs)
        '''
        if game < num_initial_observation:
            print("Observing game ", game)
        else:
            print("Learning & playing game ", game)
        '''
        for step in range (5000):

            if game < num_initial_observation:
                #take a radmon action
                a = env.action_space.sample()
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob or game%random_every_n==0:
                    #take a random action
                    a = env.action_space.sample()

                else:

                    #Get Remembered optiomal policy
                    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)

                    stock = np.zeros(9)
                    stockAction = np.zeros(shape=(9,num_env_actions))
                    for i in range(9):
                        stockAction[i] = env.action_space.sample()
                        stock[i] = predictTotalRewards(qs,stockAction[i])
                    best_index = np.argmax(stock)
                    randaction = stockAction[best_index]

                    #Compare R for SmartCrossEntropy action with remembered_optimal_policy and select the best
                    #if predictTotalRewards(qs,remembered_optimal_policy) > utility_possible_actions[best_sce_i]:
                    if predictTotalRewards(qs,remembered_optimal_policy) > predictTotalRewards(qs,randaction):
                        a = remembered_optimal_policy
                        #print(" | selecting remembered_optimal_policy ",a)
                    else:
                        a = randaction
                        #print(" - selecting generated optimal policy ",a)


            env.render()
            qs_a = np.concatenate((qs,a), axis=0)

            #get the target state and reward
            s,r,done,info = env.step(a)
            #record only the first x number of states


            if step ==0:
                gameSA[0] = qs_a
                gameS[0] = qs
                gameR[0] = np.array([r])
                gameA[0] = np.array([r])
            else:
                gameSA= np.vstack((gameSA, qs_a))
                gameS= np.vstack((gameS, qs))
                gameR = np.vstack((gameR, np.array([r])))
                gameA = np.vstack((gameA, np.array([a])))

            if step > 800:
                done = True

            if done :
                tempGameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
                tempGameS = np.zeros(shape=(1,num_env_variables))
                tempGameA = np.zeros(shape=(1,num_env_actions))
                tempGameR = np.zeros(shape=(1,1))
                tempGameRR = np.zeros(shape=(1,1))

                #Calculate Q values from end to start of game
                #mstats.append(step)
                for i in range(0,gameR.shape[0]):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]
                    else:
                        #print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]+b_discount*gameR[(gameR.shape[0]-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])
                    if i==gameR.shape[0]-1 and game%5==0:
                        print("Training Game #",game,"last everage",memoryR[:-1000].mean(),"game mean",gameR.mean(),"memoryR",memoryR.shape[0], "memoryRR ",memoryRR.shape[0],"memoryRR mean",memoryRR.mean(axis=0)[0], " steps = ", step ,"last reward", r," finished with headscore ", gameR[(gameR.shape[0]-1)-i][0])

                if memoryR.shape[0] ==1:
                    memorySA = gameSA
                    memoryR = gameR
                    memoryA = gameA
                    memoryS = gameS
                    memoryRR = gameR





                tempGameA = tempGameA[1:]
                tempGameS = tempGameS[1:]
                tempGameRR = tempGameRR[1:]
                tempGameR = tempGameR[1:]
                tempGameSA = tempGameSA[1:]


                for i in range(gameR.shape[0]):
                    tempGameSA = np.vstack((tempGameSA,gameSA[i]))
                    tempGameR = np.vstack((tempGameR,gameR[i]))


                    #Add experience to memory
                    #memorySA = np.concatenate((memorySA,gameSA),axis=0)
                    #memoryR = np.concatenate((memoryR,gameR),axis=0)

                #print("memoryR average", memoryR.mean(axis=0)[0])
                for i in range(0,gameR.shape[0]):
                    if game > 15 and addToMemory(gameR[i][0],memoryRR.mean(axis=0)[0],memoryRR.max()):
                        tempGameA = np.vstack((tempGameA,gameA[i]))
                        tempGameS = np.vstack((tempGameS,gameS[i]))
                        tempGameRR = np.vstack((tempGameRR,gameR[i]))




                if memoryR.shape[0] ==1:
                    memoryA = tempGameA
                    memoryS = tempGameS
                    memoryRR = tempGameRR
                    memoryR = tempGameR
                    memorySA = tempGameSA
                else:
                    #Add experience to memory
                    memoryS = np.concatenate((memoryS,tempGameS),axis=0)
                    memoryRR = np.concatenate((memoryRR,tempGameRR),axis=0)
                    memoryA = np.concatenate((memoryA,tempGameA),axis=0)
                    memorySA = np.concatenate((memorySA,tempGameSA),axis=0)
                    memoryR = np.concatenate((memoryR,tempGameR),axis=0)



                #if memory is full remove first element
                if np.alen(memoryR) >= max_memory_len:
                    memorySA = memorySA[gameR.shape[0]:]
                    memoryR = memoryR[gameR.shape[0]:]
                if np.alen(memoryA) >= max_memory_len/10:
                    memoryA = memoryA[gameR.shape[0]:]
                    memoryS = memoryS[gameR.shape[0]:]
                    memoryRR = memoryRR[gameR.shape[0]:]


            #Update the states
            qs=s


            #Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game%2 == 0:
                    if game%25 == 0:
                        print("Training  game# ", game,"momory size", memorySA.shape[0])



                    tSA = (memorySA)
                    tR = (memoryR)
                    tX = (memoryS)
                    tY = (memoryA)
                    #sw = (memoryAdv)
                    train_Q = np.random.randint(tR.shape[0],size=experience_replay_size)
                    train_A = np.random.randint(tY.shape[0],size=experience_replay_size)

                    print("train_Q", train_Q)
                    print("train_A", train_A)

                    tX = tX[train_A,:]
                    tY = tY[train_A,:]
                    #sw = sw[train_idx,:]
                    tR = tR[train_Q,:]
                    tSA = tSA[train_Q,:]
                    #training Reward predictor model
                    Qmodel.fit(tSA,tR, batch_size=mini_batch,epochs=training_epochs,verbose=0)

                    #training action predictor model
                    action_predictor_model.fit(tX,tY, batch_size=mini_batch, epochs=training_epochs,verbose=0)

            if done and game >= num_initial_observation:
                if save_weights and game%20 == 0 and game >35:
                    #Save model
                    #print("Saving weights")
                    Qmodel.save_weights(weigths_filename)
                    action_predictor_model.save_weights(apWeights_filename)

                    #np.save('memorySA.npy',memorySA)
                    #np.save('memoryRR.npy',memoryRR)
                    #np.save('memoryS.npy',memoryS)
                    #np.save('memoryA.npy',memoryA)
                    #np.save('memoryR.npy',memoryR)




            if done:
                '''
                #Game won  conditions
                if step > 197:
                    print("Game ", game," WON *** " )
                else:
                    print("Game ",game," ended with positive reward ")
                #Game ended - Break
                '''
                break


plt.plot(mstats)
plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    Qmodel.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
