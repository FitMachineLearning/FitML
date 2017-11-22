'''
BipedWalker solution by Michel Aka
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Using Selective Memory
'''
import numpy as np
import keras
import gym
import os
import h5py
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


num_env_variables = 24
num_env_actions = 4
num_initial_observation = 100
learning_rate =  0.003
apLearning_rate = 0.003
weigths_filename = "BipedWalker-SM-weights.h5"
apWeights_filename = "BipedWalker-SM-weights-ap.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.95
max_memory_len = 6000
starting_explore_prob = 0.15
training_epochs = 8
load_previous_weights = False
observe_and_train = True
save_weights = True
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
model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(1024, activation='relu', input_dim=dataX.shape[1]))
model.add(Dense(256, activation='tanh'))
model.add(Dense(dataY.shape[1]))

opt = optimizers.adam(lr=learning_rate)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(1024, activation='relu', input_dim=apdataX.shape[1]))
action_predictor_model.add(Dense(512, activation='relu'))
action_predictor_model.add(Dense(apdataY.shape[1],activation='tanh'))

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

memorySA = np.zeros(shape=(1,num_env_variables+num_env_actions))
memoryS = np.zeros(shape=(1,num_env_variables))
memoryA = np.zeros(shape=(1,1))
memoryR = np.zeros(shape=(1,1))

mstats = []



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


def addToMemory(reward,rangeL,rangeH):
    prob = reward - rangeL
    prob = prob / (rangeH - rangeL)
    if np.random.rand(1)<=prob :
        print("Adding reward",reward," based on prob ", prob)
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
        if game < num_initial_observation:
            print("Observing game ", game)
        else:
            print("Learning & playing game ", game)
        for step in range (5000):

            if game < num_initial_observation:
                #take a radmon action
                a = env.action_space.sample()
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob:
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
                gameA[0] = a
            else:
                gameSA= np.vstack((gameSA, qs_a))
                gameS= np.vstack((gameS, qs))
                gameR = np.vstack((gameR, np.array([r])))
                gameA = np.vstack((gameA, a))


            if done :
                tempGameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
                tempGameS = np.zeros(shape=(1,num_env_variables))
                tempGameA = np.zeros(shape=(1,num_env_actions))
                tempGameR = np.zeros(shape=(1,1))
                #Calculate Q values from end to start of game
                mstats.append(step)
                for i in range(0,gameR.shape[0]):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]
                    else:
                        #print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]+b_discount*gameR[(gameR.shape[0]-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])
                    #print("reward ", gameR[(gameR.shape[0]-1)-i][0])
                    if i==gameR.shape[0]-1:
                        print("Training Game #",game,"memory size ",memoryR.shape[0], " steps = ", step ,"last reward", r," finished with headscore ", gameR[(gameR.shape[0]-1)-i][0])

                #selective Memeory
                for i in range(0,gameR.shape[0]):
                    if addToMemory(gameR[i][0],-10,20):
                        tempGameSA = np.vstack((tempGameSA, gameSA[i]))
                        tempGameA = np.vstack((tempGameA,gameA[i]))
                        tempGameR = np.vstack((tempGameR,gameR[i]))
                        tempGameS = np.vstack((tempGameS,gameS[i]))

                #remove zeros
                tempGameSA = tempGameSA[1:]
                tempGameA = tempGameA[1:]
                tempGameS = tempGameS[1:]
                tempGameR = tempGameR[1:]


                if memoryR.shape[0] ==1:
                    memorySA = tempGameSA
                    memoryR = tempGameR
                    memoryA = tempGameA
                    memoryS = tempGameS
                else:
                    #Add experience to memory
                    memorySA = np.concatenate((memorySA,tempGameSA),axis=0)
                    memoryS = np.concatenate((memoryS,tempGameS),axis=0)
                    memoryR = np.concatenate((memoryR,tempGameR),axis=0)
                    memoryA = np.concatenate((memoryA,tempGameA),axis=0)


                #if memory is full remove first element
                if np.alen(memorySA) >= max_memory_len:
                    memoryR = memoryR[np.alen(tempGameR):]
                    memorySA = memorySA[np.alen(tempGameR):]
                    memoryS = memoryS[np.alen(tempGameR):]
                    memoryA = memoryA[np.alen(tempGameR):]



            #Update the states
            qs=s

            if step > 1998:
                done = True
            #Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game%10 == 0:
                    print("Training  game# ", game,"momory size", memorySA.shape[0])

                    #training Reward predictor model
                    model.fit(memorySA,memoryR, batch_size=32,epochs=training_epochs,verbose=0)

                    #training action predictor model
                    action_predictor_model.fit(memoryS,memoryA, batch_size=32, epochs=training_epochs,verbose=0)

            if done and game >= num_initial_observation:
                if save_weights and game%20 == 0:
                    #Save model
                    print("Saving weights")
                    model.save_weights(weigths_filename)
                    action_predictor_model.save_weights(apWeights_filename)

            if done:
                #Game won  conditions
                if step > 197:
                    print("Game ", game," WON *** " )
                else:
                    print("Game ",game," ended with positive reward ")
                #Game ended - Break
                break


plt.plot(mstats)
plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
