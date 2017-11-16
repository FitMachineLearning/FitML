'''
CartPole solution by Michel Aka

https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Using Actor Critic
Note that I prefe the terms Action Predictor Network and Q/Reward Predictor network better

Update
Cleaned up variables and more readable memory
Improved hyper parameters for better performance

'''
import numpy as np
import keras
import gym
import os
import h5py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


num_env_variables = 4
num_env_actions = 1
num_initial_observation = 10
learning_rate =  0.001
apLearning_rate = 0.003
weigths_filename = "CartPole-HRM-v2-weights.h5"
apWeights_filename = "CartPole_HRM-QL-v2-weights.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.98
max_memory_len = 10000
starting_explore_prob = 0.05
training_epochs = 5
load_previous_weights = False
observe_and_train = True
save_weights = True
num_games_to_play = 500


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('CartPole-v0')
env.reset()




#nitialize the Reward predictor model
model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(128, activation='relu', input_dim=num_env_variables+num_env_actions))
#outputs a reward value
model.add(Dense(1))

opt = optimizers.adam(lr=learning_rate)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(128, activation='relu', input_dim=num_env_variables))
action_predictor_model.add(Dense(num_env_actions))
opt2 = optimizers.adam(lr=apLearning_rate)
action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])

# initialize the action state reward matcher
# action_sate_reward_matcher(s,R)->a
# remembers which action to take in order to get a specific reward
action_sate_reward_matcher = Sequential()
action_sate_reward_matcher.add(Dense(128, activation='relu', input_dim=num_env_variables+1))
action_sate_reward_matcher.add(Dense(num_env_actions))
opt2 = optimizers.adam(lr=learning_rate)
action_sate_reward_matcher.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])




# initialize the highest reward memory model
# hightest_reward_memory(s)->R
# it remembers the highest Reward (expected sum of discounted rewards) for this specific state
# This network will be initialized to remember only negative events. It will then update its weights based on experience
highest_reward_memory_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
highest_reward_memory_model.add(Dense(1024, activation='tanh', input_dim=num_env_variables))
highest_reward_memory_model.add(Dense(512, activation='tanh'))
highest_reward_memory_model.add(Dense(1))
opt2 = optimizers.adam(lr=apLearning_rate*3)
highest_reward_memory_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])


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

#StateAction array
memorySA = np.zeros(shape=(1,num_env_variables+num_env_actions))
#State
memoryS = np.zeros(shape=(1,num_env_variables))
#StateHighestReward array
memorySHR = np.zeros(shape=(1,num_env_variables+1))
#Action array
memoryA = np.zeros(shape=(1,1))
#Value/Reward array
memoryR = np.zeros(shape=(1,1))
#Highest Value/Reward array
memoryHR = np.zeros(shape=(1,1))
#Best Action array
memoryBA = np.zeros(shape=(1,1))

mstats = []


def initilizeHighestRewardMemory():
    dataS = np.random.rand(20,num_env_variables)
    dataR = np.full((20,1),-1)
    highest_reward_memory_model.fit(dataS,dataR, batch_size=32, epochs=5,verbose=0)


def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = model.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward

def GetActionForThisStateReward(qstate,R):

    predX = np.zeros(shape=(1,num_env_variables+1))
    #print("predX",predX)
    predX[0] = np.concatenate( (qstate, np.array([R])) , axis=0)


    pred = action_sate_reward_matcher.predict(predX[0].reshape(1,predX.shape[1]))
    action4StateReward = pred[0][0]
    return action4StateReward

def GetHighestRewardForState(qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = highest_reward_memory_model.predict(predX[0].reshape(1,predX.shape[1]))
    highest_remembered_Reward = pred[0][0]
    #print ("highest_remembered_Reward",highest_remembered_Reward)
    return highest_remembered_Reward

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


initilizeHighestRewardMemory()

'''
for i in range(10):
    res = highest_reward_memory_model.predict(np.random.rand(1,num_env_variables))
    print("highest_reward_memory_model", res)

for i in range(10):
    dataS = np.random.rand(1,num_env_variables)
    dataR = np.full((1,1),-10+i)
    highest_reward_memory_model.fit(dataS,dataR, epochs=3,verbose=2)
    res = highest_reward_memory_model.predict(dataS)
    print("highest_reward_memory_model", res)
'''

if observe_and_train:

    #Play the game 500 times
    for game in range(num_games_to_play):
        gameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
        gameS = np.zeros(shape=(1,num_env_variables))
        gameA = np.zeros(shape=(1,1))
        gameBA = np.zeros(shape=(1,1))
        gameR = np.zeros(shape=(1,1))
        gameHR = np.zeros(shape=(1,1))
        gameSHR =  np.zeros(shape=(1,num_env_variables+1))

        #Get the Q state
        qs = env.reset()
        #print("qs ", qs)
        if game < num_initial_observation:
            print("Observing game ", game)
        else:
            print("Learning & playing game ", game)
        for step in range (500):
            highest_reward = GetHighestRewardForState(qs)
            best_action = 0
            if game < num_initial_observation:
                #take a radmon action
                a = np.array([env.action_space.sample()])
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob:
                    #take a random action
                    a=np.array([env.action_space.sample()])

                else:
                    # Get highest remembered reward for this state


                    action4StateReward = GetActionForThisStateReward(qs,highest_reward)
                    best_action = action4StateReward

                    action4StateReward = np.array([action4StateReward])

                    #Get Remembered optiomal policy
                    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)

                    #print("action4StateReward", action4StateReward,"remembered_optimal_policy", remembered_optimal_policy)


                    if predictTotalRewards(qs,remembered_optimal_policy) > predictTotalRewards(qs,action4StateReward):
                        action4StateReward = remembered_optimal_policy


                    randaction = np.array([env.action_space.sample()])

                    #print("highest_reward", highest_reward)
                    #mstatsR.append(highest_reward)

                    #Compare R for SmartCrossEntropy action with remembered_optimal_policy and select the best
                    #if predictTotalRewards(qs,remembered_optimal_policy) > utility_possible_actions[best_sce_i]:
                    if predictTotalRewards(qs,action4StateReward) > predictTotalRewards(qs,randaction):
                        a = action4StateReward
                        #print(" | selecting remembered_optimal_policy ",a)
                    else:
                        a = randaction
                        #print(" - selecting generated optimal policy ",a)


            if a[0] <0:
                a [0]= 0
            if a[0] > 1:
                a[0] = 1

            env.render()
            a = np.around(a)
            a = a.astype(int)
            qs_a = np.concatenate((qs,a), axis=0)

            #get the target state and reward
            s,r,done,info = env.step(a[0])
            #record only the first x number of states

            if done and step < 197:
                r=-1

            if step ==0:
                gameSA[0] = qs_a
                gameS[0] = qs
                gameR[0] = np.array([r])
                gameA[0] = np.array([r])
                gameHR[0] = np.array([highest_reward])
                gameSHR[0] = np.array(np.concatenate( (qs, np.array([highest_reward])), axis=0   ))
                gameBA[0] = np.array([best_action])

            else:
                gameSA = np.vstack((gameSA, qs_a))
                gameS = np.vstack((gameS, qs))
                gameSHR = np.vstack((gameSHR,  np.concatenate( (qs, np.array([highest_reward])), axis=0   )  ))
                gameR = np.vstack((gameR, np.array([r])))
                gameA = np.vstack((gameA, np.array([a])))
                gameBA = np.vstack((gameBA, np.array([best_action])))
                gameHR = np.vstack((gameHR, np.array([highest_reward])))



            if done :
                mstats.append(step)
                #Calculate Q values from end to start of game
                for i in range(0,gameR.shape[0]):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]
                    else:
                        #print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]+b_discount*gameR[(gameR.shape[0]-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])

                    if gameR[(gameR.shape[0]-1)-i][0] > gameHR[(gameHR.shape[0]-1)-i][0]:
                        #print ("Old HR",gameHR[(gameHR.shape[0]-1)-i][0], "New HR",gameR[(gameR.shape[0]-1)-i][0] )
                        gameHR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]
                        gameSHR[(gameR.shape[0]-1)-i] = np.concatenate( (qs, np.array([highest_reward])), axis=0   )
                        gameBA[(gameR.shape[0]-1)-i][0] = gameA[(gameR.shape[0]-1)-i][0]

                    if i==gameR.shape[0]-1:
                        print("Training Game #",game, " steps = ", step ,"last reward", r,"Highest reward",gameHR[(gameHR.shape[0]-1)-i][0] ," finished with headscore ", gameR[(gameR.shape[0]-1)-i][0])

                if memoryR.shape[0] ==1:
                    memorySA = gameSA
                    memoryR = gameR
                    memoryA = gameA
                    memoryS = gameS
                    memoryHR = gameHR
                    memorySHR = gameSHR
                    memoryBA = gameBA
                else:
                    #Add experience to memory
                    memorySA = np.concatenate((memorySA,gameSA),axis=0)
                    memoryS = np.concatenate((memoryS,gameS),axis=0)
                    memoryR = np.concatenate((memoryR,gameR),axis=0)
                    memoryA = np.concatenate((memoryA,gameA),axis=0)
                    memoryBA =  np.concatenate((memoryBA,gameBA),axis=0)
                    memoryHR = np.concatenate((memoryHR,gameHR),axis=0)
                    memorySHR = np.concatenate((memorySHR,gameSHR),axis=0)


                #if memory is full remove first element
                if np.alen(memorySA) >= max_memory_len:
                    #print("memory full. mem len ", np.alen(memoryX))
                    for l in range(np.alen(gameR)):
                        memorySA = np.delete(memorySA, 0, axis=0)
                        memoryR = np.delete(memoryR, 0, axis=0)
                        memoryA = np.delete(memoryA, 0, axis=0)
                        memoryS = np.delete(memoryS, 0, axis=0)
                        memoryHR = np.delete(memoryHR, 0, axis=0)
                        memoryBA = np.delete(memoryBA, 0, axis=0)
                        memorySHR = np.delete(memorySHR, 0, axis=0)



            #Update the states
            qs=s

            if step > 497:
                done = True
            #Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game%3 == 0:
                    print("Training  game# ", game,"momory size", memorySA.shape[0])

                    #training Reward predictor model
                    model.fit(memorySA,memoryR, batch_size=128,epochs=training_epochs,verbose=0)

                    highest_reward_memory_model.fit(memoryS,memoryHR,batch_size=128,epochs=training_epochs,verbose=0)

                    action_sate_reward_matcher.fit(memorySHR,memoryBA,batch_size=128,epochs=training_epochs,verbose=0)

                    #training action predictor model
                    action_predictor_model.fit(memoryS,memoryBA, batch_size=128, epochs=training_epochs,verbose=0)

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

#plt.plot(mstatsR)
#plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
