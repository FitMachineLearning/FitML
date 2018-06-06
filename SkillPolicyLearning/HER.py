'''
**** EXPERIMENTAL - THIS DOES NOT WORK YET ****


solution by Michel Aka author of FitML github blog and repository
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Update
Deep Network
Starts to land consistantly at 350
'''

import numpy as np
import keras
import gym
#import pybullet
#import pybullet_envs
#import roboschool


import pygal
import os
import h5py
#import matplotlib.pyplot as plt
import math

from random import gauss
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


PLAY_GAME = False #Set to True if you want to agent to play without training
uses_critic = True
uses_parameter_noising = True

ENVIRONMENT_NAME = "CartPole-v0"
num_env_variables = 4
num_env_actions = 2


num_initial_observation = 30
learning_rate =  0.0002
apLearning_rate = 0.0001

MUTATION_PROB = 0.4

littl_sigma = 0.00006
big_sigma = 0.01
upper_delta = 0.075
lower_delta = 0.05
#gaussSigma = 0.01
version_name = ENVIRONMENT_NAME + "ker_v12"
weigths_filename = version_name+"-weights.h5"
apWeights_filename = version_name+"-weights-ap.h5"


#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.995
max_memory_len = 200000
experience_replay_size = 2000
num_retries = 60
starting_explore_prob = 0.05
training_epochs = 5
mini_batch = 512*4
load_previous_weights = False
observe_and_train = True
save_weights = True
save_memory_arrays = True
load_memory_arrays = False


random_every_n = 2
train_every_n_game = 2
num_games_to_play = 20000
num_games_to_test = 5
test_game_every = 30


random_num_games_to_play = num_games_to_play/3
USE_GAUSSIAN_NOISE = True
CLIP_ACTION = False
HAS_REWARD_SCALLING = False
USE_ADAPTIVE_NOISE = True
HAS_EARLY_TERMINATION_REWARD = False
EARLY_TERMINATION_REWARD = -5
max_steps = 995



#Selective memory settings
sm_normalizer = 20
sm_memory_size = 10500

last_game_average = -1000
last_best_noisy_game = -1000
max_game_average = -1000
num_positive_avg_games = 0

#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment

env = gym.make(ENVIRONMENT_NAME)
#env.render(mode="human")
env.reset()


print("-- Observations",env.observation_space)
print("-- actionspace",env.action_space)

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


#initialize the action predictor model
action_policy_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_policy_model.add(Dense(256, activation='relu', input_dim=num_env_variables*2))
#action_predictor_model.add(Dropout(0.5))
action_policy_model.add(Dense(256, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
#action_predictor_model.add(Dense(256, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
#action_predictor_model.add(Dense(64, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
action_policy_model.add(Dense(apdataY.shape[1]))
opt2 = optimizers.rmsprop(lr=apLearning_rate)
#opt2 = optimizers.Adadelta()

action_policy_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])


#initialize the action predictor model
noisy_policy_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
noisy_policy_model.add(Dense(256, activation='relu', input_dim=num_env_variables*2))
#action_predictor_model.add(Dropout(0.5))
noisy_policy_model.add(Dense(256, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
#action_predictor_model.add(Dense(256, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
#action_predictor_model.add(Dense(64, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
noisy_policy_model.add(Dense(apdataY.shape[1]))
opt2 = optimizers.rmsprop(lr=apLearning_rate)
#opt2 = optimizers.Adadelta()

noisy_policy_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])



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
        action_policy_model.load_weights(apWeights_filename)
    else:
        print("File ",apWeights_filename," does not exis. Retraining... ")


memorySA = np.zeros(shape=(1,num_env_variables+num_env_actions))
memoryS = np.zeros(shape=(1,num_env_variables))
memoryS_S1 = np.zeros(shape=(1,num_env_variables))
memoryA = np.zeros(shape=(1,1))
memoryR = np.zeros(shape=(1,1))
memoryRR = np.zeros(shape=(1,1))
memoryW = np.zeros(shape=(1,1))


if load_memory_arrays:
    if os.path.isfile(version_name+'memorySA.npy'):
        print("Memory Files exist. Loading...")
        memorySA = np.load(version_name+'memorySA.npy')
        memoryRR = np.load(version_name+'memoryRR.npy')
        memoryS = np.load(version_name+'memoryS.npy')
        memoryS_S1 = np.load(version_name+'memoryS.npy')
        memoryA = np.load(version_name+'memoryA.npy')
        memoryR = np.load(version_name+'memoryR.npy')
        memoryW = np.load(version_name+'memoryW.npy')

    else:
        print("No memory Files. Recreating")


mstats = []
mGames = []
mAverageScores = []
mSteps = []
mAP_Counts = 0
oldAPCount = 0
num_add_mem = 0
mAPPicks = []

#------

def add_noise_simple(mu,big_sigma, largeNoise=False):
    x =   np.random.rand(1) - 0.5 #probability of doing x
    if not largeNoise:
        x = x*big_sigma
    else:
        x = x*big_sigma   #Sigma = width of the standard deviaion
    #print ("x/200",x,"big_sigma",big_sigma)
    return mu + x

add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])

# --- Parameter Noising

def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = Qmodel.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward

def predictActionFromS_S1(qstate):
    predX = np.zeros(shape=(1,num_env_variables*2))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = action_policy_model.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy





#Play the game 500 times
for game in range(num_games_to_play):
    gameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
    gameS = np.zeros(shape=(1,num_env_variables))
    gameS_S1 = np.zeros(shape=(1,num_env_variables*2))
    gameA = np.zeros(shape=(1,num_env_actions))
    gameR = np.zeros(shape=(1,1))
    gameW = np.zeros(shape=(1,1))
    #Get the Q state
    qs = env.reset()
    qs_s1 = np.concatenate((qs,qs), axis=0)
    qs_1 = qs
    qs_2 = qs
    mAP_Counts = 0
    num_add_mem = 0
    #print("qs ", qs)
    is_noisy_game = False



    for step in range (5000):

        if PLAY_GAME:
            remembered_optimal_policy = predictActionFromS_S1(qs)
            a = remembered_optimal_policy
        elif game < num_initial_observation:
            #take a radmon action
            a = keras.utils.to_categorical(env.action_space.sample(),num_env_actions)
        else:
            prob = np.random.rand(1)
            explore_prob = starting_explore_prob-(starting_explore_prob/random_num_games_to_play)*game

            if game > random_num_games_to_play:
                prob = 0.000001
            #Chose between prediction and chance
            if prob < explore_prob or game%random_every_n==1:
                #take a random action
                a =  keras.utils.to_categorical(env.action_space.sample(),num_env_actions)

            else:
                #print("Using Actor")
                if is_noisy_game and uses_parameter_noising:
                    remembered_optimal_policy = GetRememberedOptimalPolicyFromNoisyModel(noisy_model,qs)
                    #mAP_Counts = oldAPCount
                else:
                    #remembered_optimal_policy = predictActionFromS_S1(qs_s1)
                    pqs_1 = qs
                    pqs_2 = [0,0,0,0]
                    pqs_2 = add_noise_simple(pqs_2,0.2,True)
                    pqs_s1 = np.concatenate((pqs_1,pqs_2), axis=0)
                    #if testStep%100==0:
                    #    print("#",game, " play qs_s1",pqs_s1)
                    remembered_optimal_policy = predictActionFromS_S1(pqs_s1)
                a = remembered_optimal_policy


        if CLIP_ACTION:
            for i in range (np.alen(a)):
                if a[i] < -1: a[i]=-0.99999999999
                if a[i] > 1: a[i] = 0.99999999999



        action = np.argmax(a)

        env.render()
        #print("a ", a, "qs", qs)
        qs_a = np.concatenate((qs,a), axis=0)
        #print("qs_a", qs_a)

        qs_1 = qs
        #get the target state and reward
        s,r,done,info = env.step(action)
        #record only the first x number of states
        qs_2 = s
        qs_s1 = np.concatenate((qs_1,qs_2), axis=0)

        #if step%100==0:
        #    print("play qs_s1",qs_s1)

        if HAS_EARLY_TERMINATION_REWARD:
            if done and step<max_steps-3:
                r = EARLY_TERMINATION_REWARD
        if  HAS_REWARD_SCALLING:
            r=r/200 #reward scalling to from [-1,1] to [-100,100]

        if step ==0:
            gameSA[0] = qs_a
            gameS[0] = qs
            gameS_S1[0] = qs_s1
            gameR[0] = np.array([r])
            gameA[0] = np.array([r])
            gameW[0] =  np.array([0.000000005])
        else:
            gameSA= np.vstack((gameSA, qs_a))
            gameS= np.vstack((gameS, qs))
            gameS_S1 = np.vstack((gameS_S1, qs_s1))
            gameR = np.vstack((gameR, np.array([r])))
            gameA = np.vstack((gameA, np.array([a])))
            gameW = np.vstack((gameW, np.array([0.000000005])))

        if step > max_steps:
            done = True

        if done :
            tempGameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
            tempGameS = np.zeros(shape=(1,num_env_variables))
            tempGameS_S1 = np.zeros(shape=(1,num_env_variables))

            tempGameA = np.zeros(shape=(1,num_env_actions))
            tempGameR = np.zeros(shape=(1,1))
            tempGameRR = np.zeros(shape=(1,1))
            tempGameW = np.zeros(shape=(1,1))

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

            if memoryR.shape[0] ==1:
                memorySA = gameSA
                memoryR = gameR
                memoryA = gameA
                memoryS = gameS
                memoryS_S1 = gameS_S1

                memoryRR = gameR
                memoryW = gameW




            if memoryR.shape[0] ==1:
                memoryA = gameA
                memoryS = gameS
                memoryS_S1 = gameS_S1

                memoryRR = gameRR
                memoryR = gameR
                memorySA = gameSA
                memoryW = gameW
            else:
                #Add experience to memory
                memoryS = np.concatenate((memoryS,gameS),axis=0)
                memoryS_S1 = np.concatenate((memoryS_S1,gameS_S1),axis=0)

                #memoryRR = np.concatenate((memoryRR,gameRR),axis=0)
                memoryA = np.concatenate((memoryA,gameA),axis=0)
                memorySA = np.concatenate((memorySA,gameSA),axis=0)

                memoryR = np.concatenate((memoryR,gameR),axis=0)
                memoryW = np.concatenate((memoryW,gameW),axis=0)


                if gameR.mean() > max_game_average :
                    max_game_average = gameR.mean()

            #if memory is full remove first element
            if np.alen(memoryR) >= max_memory_len:
                memorySA = memorySA[gameR.shape[0]:]
                memoryR = memoryR[gameR.shape[0]:]
                memoryA = memoryA[gameR.shape[0]:]
                memoryS = memoryS[gameR.shape[0]:]
                memoryS_S1 = memoryS_S1[gameR.shape[0]:]
                #memoryRR = memoryRR[gameR.shape[0]:]
                memoryW = memoryW[gameR.shape[0]:]


        qs=s

        if done and game > num_initial_observation and not PLAY_GAME:
            last_game_average = gameR.mean()
            if is_noisy_game and last_game_average > memoryR.mean():
                last_best_noisy_game = last_game_average
            #if game >3:
                #actor_experience_replay(gameSA,gameR,gameS,gameA,gameW,1)


            if game > 3 and game %train_every_n_game ==0:
                for t in range(training_epochs):
                    print("Experience replay ")
                    #tSA = (memorySA)
                    tS_S1 = (memoryS_S1)
                    tA = (memoryA)
                    #tR = (memoryR)
                    train_A = np.random.randint(tA.shape[0],size=int(min(experience_replay_size,np.alen(tA) )))
                    tA = tA[train_A,:]
                    tS_S1 = tS_S1[train_A,:]
                    #print("Training Critic n elements =", np.alen(tR))
                    action_policy_model.fit(tS_S1,tA, batch_size=mini_batch, nb_epoch=1,verbose=0)



        if done and game >= num_initial_observation and not PLAY_GAME:
            if save_weights and game%20 == 0 and game >35:
                #Save model
                #print("Saving weights")
                #Qmodel.save_weights(weigths_filename)
                action_policy_model.save_weights(apWeights_filename)

            if save_memory_arrays and game%20 == 0 and game >35:
                np.save(version_name+'memorySA.npy',memorySA)
                np.save(version_name+'memoryRR.npy',memoryRR)
                np.save(version_name+'memoryS.npy',memoryS)
                np.save(version_name+'memoryA.npy',memoryA)
                np.save(version_name+'memoryR.npy',memoryR)
                np.save(version_name+'memoryW.npy',memoryW)




        if done:
            oldAPCount = mAP_Counts
            if gameR.mean() >0:
                num_positive_avg_games += 1
            if game%1==0:
                #print("Training Game #",game,"last everage",memoryR.mean(),"max_game_average",max_game_average,,"game mean",gameR.mean(),"memMax",memoryR.max(),"memoryR",memoryR.shape[0], "SelectiveMem Size ",memoryRR.shape[0],"Selective Mem mean",memoryRR.mean(axis=0)[0], " steps = ", step )
                if is_noisy_game:
                    print("Noisy Game #  %7d  avgScore %8.3f  last_game_avg %8.3f  max_game_avg %8.3f  memory size %8d memMax %8.3f steps %5d pos games %5d" % (game, memoryR.mean(), last_game_average, memoryW.max() , memoryR.shape[0], memoryR.max(), step,num_positive_avg_games    ) )
                else:
                    print("Reg Game   #  %7d  avgScore %8.3f  last_game_avg %8.3f  max_game_avg %8.3f  memory size %8d memMax %8.3f steps %5d pos games %5d" % (game, memoryR.mean(), last_game_average, memoryW.max() , memoryR.shape[0], memoryR.max(), step ,num_positive_avg_games   ) )

            if game%5 ==0 and np.alen(memoryR)>1000:
                mGames.append(game)
                mSteps.append(step/1000*100)
                mAPPicks.append(mAP_Counts/step*100)

                mAverageScores.append(max(memoryR.mean()*200, -150))


                bar_chart = pygal.HorizontalLine()
                bar_chart.x_labels = map(str, mGames)                                            # Then create a bar graph object
                bar_chart.add('Average score', mAverageScores)  # Add some values
                bar_chart.add('percent actor picks ', mAPPicks)  # Add some values
                bar_chart.add('percent steps complete ', mSteps)  # Add some values


                bar_chart.render_to_file(version_name+'Performance2_bar_chart.svg')


        ''' TEST GAMES '''

        if done and game%test_game_every==0:
            for testGame in range (num_games_to_test):
                qs = env.reset()
                qs_s1 = np.concatenate((qs,qs), axis=0)
                print("# ",testGame)
                for testStep in range (max_steps):
                    qs_1 = qs
                    qs_2 = [0,0,0,0]
                    qs_s1 = np.concatenate((qs_1,qs_2), axis=0)
                    #if testStep%100==0:
                    #    print("#",testGame, " test qs_s1",qs_s1)
                    action = predictActionFromS_S1(qs_s1)
                    #get the target state and reward
                    action = np.argmax(action)
                    s,r,testDone,info = env.step(action)
                    qs = s
                    env.render()
                    if testDone:
                        qs = env.reset()
                        break


        if done:
            break


# PLAY_GAME,



if save_weights:
    #Save model
    print("Saving weights")
    Qmodel.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
