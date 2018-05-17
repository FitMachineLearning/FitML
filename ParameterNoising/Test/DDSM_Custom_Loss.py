'''
LunarLanderContinuous solution with
 - Selective Memory
 - Actor Critic
 - Parameter Noising
 - Q as discriminator
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
import keras.backend as K


PLAY_GAME = False #Set to True if you want to agent to play without training
uses_critic = True
uses_parameter_noising = True

ENVIRONMENT_NAME = "LunarLanderContinuous-v2"
num_env_variables = 8
num_env_actions = 2


num_initial_observation = 0
learning_rate =  0.02
apLearning_rate = 0.01

MUTATION_PROB = 0.4

littl_sigma = 0.00006
big_sigma = 0.01
upper_delta = 0.075
lower_delta = 0.05
#gaussSigma = 0.01
version_name = ENVIRONMENT_NAME + "ker_v11"
weigths_filename = version_name+"-weights.h5"
apWeights_filename = version_name+"-weights-ap.h5"


#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.98
max_memory_len = 100000
experience_replay_size = 2500
random_every_n = 50
num_retries = 60
starting_explore_prob = 0.005
training_epochs = 2
mini_batch = 512*4
load_previous_weights = False
observe_and_train = True
save_weights = True
save_memory_arrays = True
load_memory_arrays = False
do_training = True
num_games_to_play = 20000
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
env.render(mode="human")
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

def dice_coef(y_true, y_pred, averageReward, predReward):
    step_losses = (y_true - y_pred) * math.fabs(averageReward-predReward)
    return K.sum(step_losses)

def dice_loss( averageReward, predReward):
    def dice(y_true, y_pred):
        return -dice_coef(y_true, y_pred,  averageReward, predReward)
    return dice

#nitialize the Reward predictor model
Qmodel = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(256, activation='relu', input_dim=dataX.shape[1]))
#Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(256, activation='relu'))
#Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(256, activation='relu'))
#Qmodel.add(Dropout(0.5))
#Qmodel.add(Dense(64, activation='relu'))
#Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(dataY.shape[1]))
opt = optimizers.rmsprop(lr=learning_rate)
#opt = optimizers.Adadelta()

Qmodel.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(256, activation='relu', input_dim=apdataX.shape[1]))
#action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(256, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(256, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
#action_predictor_model.add(Dense(64, activation='relu'))
#action_predictor_model.add(Dropout(0.5))

action_predictor_model.add(Dense(apdataY.shape[1]))
opt2 = optimizers.rmsprop(lr=apLearning_rate)
#opt2 = optimizers.Adadelta()

model_dice = dice_loss(averageReward=0.5, predReward=0.5)

action_predictor_model.compile(loss=model_dice, optimizer=opt2, metrics=['accuracy'])


#initialize the action predictor model
noisy_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
noisy_model.add(Dense(256, activation='relu', input_dim=apdataX.shape[1]))
#noisy_model.add(Dropout(0.5))
noisy_model.add(Dense(256, activation='relu'))
#noisy_model.add(Dropout(0.5))
noisy_model.add(Dense(256, activation='relu'))
#noisy_model.add(Dropout(0.5))
#noisy_model.add(Dense(64, activation='relu'))
#noisy_model.add(Dropout(0.5))
noisy_model.add(Dense(apdataY.shape[1]))
opt3 = optimizers.Adadelta()

noisy_model.compile(loss='mse', optimizer=opt3, metrics=['accuracy'])


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
memoryW = np.zeros(shape=(1,1))

BestGameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
BestGameS = np.zeros(shape=(1,num_env_variables))
BestGameA = np.zeros(shape=(1,num_env_actions))
BestGameR = np.zeros(shape=(1,1))
BestGameW = np.zeros(shape=(1,1))

if load_memory_arrays:
    if os.path.isfile(version_name+'memorySA.npy'):
        print("Memory Files exist. Loading...")
        memorySA = np.load(version_name+'memorySA.npy')
        memoryRR = np.load(version_name+'memoryRR.npy')
        memoryS = np.load(version_name+'memoryS.npy')
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


# --- Parameter Noising
def add_noise(mu, largeNoise=False):

    if largeNoise:
        sig = big_sigma
    else:
        #print("Adding Large parameter noise")
        sig = littl_sigma #Sigma = width of the standard deviaion
    #mu = means
    x =   np.random.rand(1) #probability of doing x
    #print ("x prob ",x)
    if x >0.5:
        return mu + np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    else:
        return mu - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# --- Parameter Noising
def add_noise_simple(mu, largeNoise=False):
    x =   np.random.rand(1) - 0.5 #probability of doing x
    if not largeNoise:
        x = x*big_sigma
    else:
        x = x*big_sigma   #Sigma = width of the standard deviaion
    #print ("x/200",x,"big_sigma",big_sigma)
    return mu + x

def add_gaussian_noise(mu,noiseSigma,largeNoise=False):
    #print ( gauss(mu, noiseSigma) )
    if np.random.rand(1) < MUTATION_PROB:
        return gauss(mu, noiseSigma)
    else:
        return mu+0.0

#add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])
add_noise = np.vectorize(add_noise,otypes=[np.float])
add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])
add_gaussian_noise = np.vectorize(add_gaussian_noise,otypes=[np.float])



def add_noise_to_model(targetModel,largeNoise = False):
    #noisy_model = keras.models.clone_model(action_predictor_model)
    #noisy_model.set_weights(action_predictor_model.get_weights())
    #print("Adding Noise to actor")
    #largeNoise =  last_game_average < memoryR.mean()
    sz = len(noisy_model.layers)
    #if largeNoise:
    #    print("Setting Large Noise!")
    for k in range(sz):
        w = targetModel.layers[k].get_weights()
        if np.alen(w) >0 :
            #print("k==>",k)
            if USE_GAUSSIAN_NOISE:
                w[0] = add_gaussian_noise(w[0],big_sigma,largeNoise)
            else:
                w[0] = add_noise_simple(w[0],largeNoise)

        targetModel.layers[k].set_weights(w)
    return targetModel




def reset_noisy_model_weights_to_apWeights(mu):
    x =  mu+0.0 #probability of doing x
    return x

reset_noisy_model_weights_to_apWeights = np.vectorize(reset_noisy_model_weights_to_apWeights,otypes=[np.float])

def reset_noisy_model():
    sz = len(noisy_model.layers)
    #if largeNoise:
    #    print("Setting Large Noise!")
    for k in range(sz):
        w = noisy_model.layers[k].get_weights()
        apW = action_predictor_model.layers[k].get_weights()

        if np.alen(w) >0:
            w[0] = reset_noisy_model_weights_to_apWeights(apW[0])
        noisy_model.layers[k].set_weights(w)
        #print("w",w)
        #print("apW",apW)

def reset_noisy_model2():
    action_predictor_model.save_weights(apWeights_filename)

    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+ apWeights_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        noisy_model.load_weights(apWeights_filename)
    else:
        print("File ",apWeights_filename," does not exis. Retraining... ")

# --- Parameter Noising

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

def GetRememberedOptimalPolicyFromNoisyModel(targetModel,qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = targetModel.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy

def addToMemory(reward,mem_mean,memMax,averegeReward,gameAverage,mstd):
    target = mem_mean + math.fabs((memMax-mem_mean)/2)
    d_target_max = math.fabs(memMax-target)
    d_target_reward = math.fabs(reward-target)
    advantage = d_target_reward / d_target_max
    gameAdvantage = math.fabs((averegeReward-gameAverage)/(averegeReward-memMax))
    prob = 0.000000000000005
    if gameAdvantage < 0.05:
        gameAdvantage = 0.000000000000005
    if reward > target:
        return True, 0.0000000005 + (1-0.0000000005)*advantage #*gameAdvantage
    else:
        return False, 0.000000000000005


def scale_weights(memR,memW):
    rmax = memR.max()
    rmin = memR.min()
    reward_range = math.fabs(rmax - rmin )
    if reward_range == 0:
        reward_range = 10
    for i in range(np.alen(memR)):
        memW[i][0] = math.fabs(memR[i][0]-rmin)/reward_range
        memW[i][0] = max(memW[i][0],0.001)
        #print("memW %5.2f reward %5.2f rmax %5.2f rmin %5.2f "%(memW[i][0],memR[i][0],rmax,rmin))
    #print("memW",memW)
    return memW


def pr_actor_experience_replay(memSA,memR,memS,memA,memW,num_epochs=1):
    for t in range(training_epochs):
        tSA = (memSA)+0.0
        tR = (memR)+0.0
        tX = (memS)+0.0
        tY = (memA)+0.0
        tW = (memW)+0.0
        tS = memW +0.0

        treshold = memoryR.mean()
        gameAverage = memoryR.mean()
        gameDistance = math.fabs(memoryW.max() - memoryR.mean())
        gameTreshold = memoryW.mean() + gameDistance*0.4

        #print("gameMean",tS.mean(),"gameMax",tS.max(),"gameTreshold",gameTreshold)

        train_C = np.arange(np.alen(tR))
        train_C = train_C[tR.flatten()>treshold]
        tX = tX[train_C,:]
        tY = tY[train_C,:]
        tW = tW[train_C,:]
        tR = tR[train_C,:]
        tS = tS[train_C,:]


        train_A = np.random.randint(tY.shape[0],size=int(min(experience_replay_size,np.alen(tR) )))

        tX = tX[train_A,:]
        tY = tY[train_A,:]
        tW = tW[train_A,:]
        tR = tR[train_A,:]
        tS = tS[train_A,:]

        tX_train = np.zeros(shape=(1,num_env_variables))
        tY_train = np.zeros(shape=(1,num_env_actions))
        for i in range(np.alen(tR)):
            pr = predictTotalRewards(tX[i],GetRememberedOptimalPolicy(tX[i]))
            #print ("tR[i]",tR[i],"pr",pr)
            d = math.fabs( memoryR.max() - pr)
            tW[i]= 0.0000000000000005
            if (tR[i]>pr ):
                tW[i]=0.15
            #if (tR[i]>pr and tS[i]>gameAverage):
            #    tW[i]=0.25
            if (tR[i]>pr + d*0.0 ):
                tW[i]=1
            #if (tR[i]>pr+d*0.005 and tR[i]>game_max) :
            #    tW[i] = 1
            if tW[i]> np.random.rand(1):
                tX_train = np.vstack((tX_train,tX[i]))
                tY_train = np.vstack((tY_train,tY[i]))


        tX_train = tX_train[1:]
        tY_train = tY_train[1:]
        print("%8d were better After removing first element"%np.alen(tX_train))
        if np.alen(tX_train)>0:
            action_predictor_model.fit(tX_train,tY_train, batch_size=mini_batch, nb_epoch=1,verbose=0)




def actor_experience_replay(memSA,memR,memS,memA,memW,num_epochs=1):
    for t in range(training_epochs):
        tSA = (memSA)+0.0
        tR = (memR)+0.0
        tX = (memS)+0.0
        tY = (memA)+0.0
        tW = (memW)+0.0
        tS = memW +0.0

        stdDev = np.std(tR)
        gameStdDev = np.std(tS)

        distance = math.fabs(memoryR.max()-memoryR.mean())
        #treshold = memoryR.mean()+ distance*0.75
        treshold = memoryR.mean()+ stdDev*1
        gameAverage = memoryR.mean()
        gameDistance = math.fabs(memoryW.max() - memoryR.mean())
        gameTreshold = memoryW.mean() + gameStdDev*0

        #print("gameMean",tS.mean(),"gameMax",tS.max(),"gameTreshold",gameTreshold)

        train_C = np.arange(np.alen(tR))
        #train_C = train_C[tS.flatten()> gameTreshold] # Only take games that are above gameTreshold
        tX = tX[train_C,:]
        tY = tY[train_C,:]
        tW = tW[train_C,:]
        tR = tR[train_C,:]
        tS = tS[train_C,:]



        #ßprint("TY.shape",tY.shape[0], "exp replay",experience_replay_size,"np ", np.alen(tR))

        if np.alen(tR) <= 0:
            break
        train_A = np.random.randint(tY.shape[0],size=int(min(experience_replay_size,np.alen(tR) )))

        tX = tX[train_A,:]
        tY = tY[train_A,:]
        tW = tW[train_A,:]
        tR = tR[train_A,:]
        tS = tS[train_A,:]


        #print("gameMean",tS.mean(),"gameMax",tS.max(),"gameTreshold",gameTreshold)

        train_D = np.arange(np.alen(tR))
        train_D = train_D[tR.flatten()>treshold] # Only take steps with rewards above threshold
        tX = tX[train_D,:]
        tY = tY[train_D,:]
        tW = tW[train_D,:]
        tR = tR[train_D,:]
        tS = tS[train_D,:]


        tX_train = tX
        tY_train = tY
        if t%89==1:
            print("%8d were better After removing first element"%np.alen(tX_train), "Upper_cut",memoryR.mean()+stdDev,"gameStdDev",memoryW.mean()+gameStdDev)
        if np.alen(tX_train)>0:
            action_predictor_model.fit(tX_train,tY_train, batch_size=mini_batch, nb_epoch=2,verbose=0)



def train_noisy_actor():
    tX = (memoryS)
    tY = (memoryA)
    tW = (memoryW)

    train_A = np.random.randint(tY.shape[0],size=int(min(experience_replay_size,np.alen(tY) )))
    tX = tX[train_A,:]
    tY = tY[train_A,:]
    tW = tW[train_A,:]

    noisy_model.fit(tX,tY, batch_size=mini_batch, nb_epoch=training_epochs,verbose=0)


def add_controlled_noise(targetModel,big_sigma,largeNoise = False):
    tR = (memoryR)
    tX = (memoryS)
    tY = (memoryA)
    tW = (memoryW)
    train_C = np.random.randint(tY.shape[0],size=100)

    tX = tX[train_C,:]
    tY_old = tY[train_C,:]
    tY_new = tY[train_C,:]
    diffs = np.zeros(np.alen(tX))
    delta = 1000
    deltaCount = 0

    for i in range(np.alen(tX)):
        a = GetRememberedOptimalPolicyFromNoisyModel(noisy_model,tX[i])
        a = a.flatten()
    #print("Output Before noise ",a)

    while ( delta > upper_delta or delta < lower_delta) and deltaCount <1:
        #noisy_model.set_weights(action_predictor_model.get_weights())
        reset_noisy_model()
        targetModel = noisy_model
        targetModel = add_noise_to_model(noisy_model,largeNoise)


        for i in range(np.alen(tX)):
            b = GetRememberedOptimalPolicyFromNoisyModel(targetModel,tX[i])
            b = b.flatten()

        c = np.abs(a-b)
        delta = c.mean()

        deltaCount+=1
        if USE_ADAPTIVE_NOISE:
            if delta > upper_delta:
                big_sigma = big_sigma *0.9
                #print("Delta",delta," out of bound adjusting big_sigma", big_sigma)

            if delta < lower_delta:
                big_sigma = big_sigma *1.1
                #print("Delta",delta," out of bound adjusting big_sigma", big_sigma)
            #if delta > 3 or delta <0.01:
            #    print("Delta",delta," out of bound adjusting big_sigma", big_sigma, "to",1/delta)
            #    big_sigma = 1 / delta
    print("Tried x time ", deltaCount,"delta =", delta,"big_sigma ",big_sigma)
    return targetModel,big_sigma




#Play the game 500 times
for game in range(num_games_to_play):
    gameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
    gameS = np.zeros(shape=(1,num_env_variables))
    gameA = np.zeros(shape=(1,num_env_actions))
    gameR = np.zeros(shape=(1,1))
    gameW = np.zeros(shape=(1,1))
    #Get the Q state
    qs = env.reset()
    mAP_Counts = 0
    num_add_mem = 0
    #print("qs ", qs)
    is_noisy_game = False

    #noisy_model.set_weights(action_predictor_model.get_weights())

    #Add noise to Actor
    if game > num_initial_observation and uses_parameter_noising:
        is_noisy_game = False
        #print("Adding Noise")
        if (game%2==0 ):
            is_noisy_game = True
            if True or last_best_noisy_game < memoryR.mean() or game%6==0:
                print("Adding BIG Noise")
                #noisy_model = keras.models.clone_model(action_predictor_model)
                reset_noisy_model()
                noisy_model,big_sigma = add_controlled_noise(noisy_model,big_sigma,True)
                #last_best_noisy_game = -1000
            '''
            else:
                print("Adding Small Noise")
                #print("Not Changing weights last_best_noisy_game", last_best_noisy_game," mean ",memoryR.mean())
                reset_noisy_model()
                add_controlled_noise(noisy_model,False)
            '''

    for step in range (5000):

        if PLAY_GAME:
            remembered_optimal_policy = GetRememberedOptimalPolicy(qs)
            a = remembered_optimal_policy
        elif game < num_initial_observation:
            #take a radmon action
            a = env.action_space.sample()
        else:
            prob = np.random.rand(1)
            explore_prob = starting_explore_prob-(starting_explore_prob/random_num_games_to_play)*game

            if game > random_num_games_to_play:
                prob = 0.000001
            #Chose between prediction and chance
            if prob < explore_prob or game%random_every_n==1:
                #take a random action
                a = env.action_space.sample()

            else:
                #print("Using Actor")
                if is_noisy_game and uses_parameter_noising:
                    remembered_optimal_policy = GetRememberedOptimalPolicyFromNoisyModel(noisy_model,qs)
                    #mAP_Counts = oldAPCount
                else:
                    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)
                a = remembered_optimal_policy

                if uses_critic :
                    #print("Using critric")
                    stock = np.zeros(num_retries)
                    stockAction = np.zeros(shape=(num_retries,num_env_actions))
                    for i in range(num_retries):
                        stockAction[i] = env.action_space.sample()
                        stock[i] = predictTotalRewards(qs,stockAction[i])
                    best_index = np.argmax(stock)
                    randaction = stockAction[best_index]

                    #Compare R for SmartCrossEntropy action with remembered_optimal_policy and select the best
                    #if predictTotalRewards(qs,remembered_optimal_policy) > utility_possible_actions[best_sce_i]:
                    if predictTotalRewards(qs,remembered_optimal_policy) > predictTotalRewards(qs,randaction):
                        a = remembered_optimal_policy
                        mAP_Counts += 1
                        #print(" | selecting remembered_optimal_policy ",a)
                    else:
                        a = randaction
                        #print(" - selecting generated optimal policy ",a)

        if CLIP_ACTION:
            for i in range (np.alen(a)):
                if a[i] < -1: a[i]=-0.99999999999
                if a[i] > 1: a[i] = 0.99999999999





        env.render()
        qs_a = np.concatenate((qs,a), axis=0)

        #get the target state and reward
        s,r,done,info = env.step(a)
        #record only the first x number of states

        if HAS_EARLY_TERMINATION_REWARD:
            if done and step<max_steps-3:
                r = EARLY_TERMINATION_REWARD
        if True or HAS_REWARD_SCALLING:
            r=r/200 #reward scalling to from [-1,1] to [-100,100]

        if step ==0:
            gameSA[0] = qs_a
            gameS[0] = qs
            gameR[0] = np.array([r])
            gameA[0] = np.array([r])
            gameW[0] =  np.array([0.000000005])
        else:
            gameSA= np.vstack((gameSA, qs_a))
            gameS= np.vstack((gameS, qs))
            gameR = np.vstack((gameR, np.array([r])))
            gameA = np.vstack((gameA, np.array([a])))
            gameW = np.vstack((gameW, np.array([0.000000005])))

        if step > max_steps:
            done = True

        if done :
            tempGameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
            tempGameS = np.zeros(shape=(1,num_env_variables))
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
                memoryRR = gameR
                memoryW = gameW

            tempGameA = tempGameA[1:]
            tempGameS = tempGameS[1:]
            tempGameRR = tempGameRR[1:]
            tempGameR = tempGameR[1:]
            tempGameSA = tempGameSA[1:]
            tempGameW =  tempGameW[1:]


            for i in range(gameR.shape[0]):
                tempGameSA = np.vstack((tempGameSA,gameSA[i]))
                tempGameR = np.vstack((tempGameR,gameR[i]))

                tempGameA = np.vstack((tempGameA,gameA[i]))
                tempGameS = np.vstack((tempGameS,gameS[i]))
                tempGameRR = np.vstack((tempGameRR,gameR[i]))
                tempGameW = np.vstack((tempGameW,gameR.mean()))


            #train actor network based on last rollout
            if game>3:
                tX = (tempGameS)
                tY = (tempGameA)
                tW = (tempGameW)
                #action_predictor_model.fit(tX,tY,sample_weight=tW.flatten(), batch_size=mini_batch, nb_epoch=training_epochs,verbose=0)



            if memoryR.shape[0] ==1:
                memoryA = tempGameA
                memoryS = tempGameS
                memoryRR = tempGameRR
                memoryR = tempGameR
                memorySA = tempGameSA
                memoryW = tempGameW
            else:
                #Add experience to memory
                memoryS = np.concatenate((memoryS,tempGameS),axis=0)
                memoryRR = np.concatenate((memoryRR,tempGameRR),axis=0)
                memoryA = np.concatenate((memoryA,tempGameA),axis=0)
                memorySA = np.concatenate((memorySA,tempGameSA),axis=0)

                memoryR = np.concatenate((memoryR,tempGameR),axis=0)
                memoryW = np.concatenate((memoryW,tempGameW),axis=0)


                if gameR.mean() > max_game_average :
                    max_game_average = gameR.mean()

            #if memory is full remove first element
            if np.alen(memoryR) >= max_memory_len:
                memorySA = memorySA[gameR.shape[0]:]
                memoryR = memoryR[gameR.shape[0]:]
                memoryA = memoryA[gameR.shape[0]:]
                memoryS = memoryS[gameR.shape[0]:]
                memoryRR = memoryRR[gameR.shape[0]:]
                memoryW = memoryW[gameR.shape[0]:]


        qs=s

        if done and game > num_initial_observation and not PLAY_GAME:
            last_game_average = gameR.mean()
            if is_noisy_game and last_game_average > memoryR.mean():
                last_best_noisy_game = last_game_average
            #if game >3:
                #actor_experience_replay(gameSA,gameR,gameS,gameA,gameW,1)

            if game > 3 and game %1 ==0:
                # train on all memory
                print("Experience Replay")
                #for i in range(3):

                actor_experience_replay(memorySA,memoryR,memoryS,memoryA,memoryW,training_epochs)
            if game > 3 and game %1 ==0 and uses_critic:
                for t in range(training_epochs):
                    tSA = (memorySA)
                    tR = (memoryR)
                    train_A = np.random.randint(tR.shape[0],size=int(min(experience_replay_size,np.alen(tR) )))
                    tR = tR[train_A,:]
                    tSA = tSA    [train_A,:]
                    #print("Training Critic n elements =", np.alen(tR))
                    Qmodel.fit(tSA,tR, batch_size=mini_batch, nb_epoch=2,verbose=0)
            if game > 3 and game %5 ==-1 and uses_parameter_noising:
                print("Training noisy_actor")
                train_noisy_actor()
                #Reinforce training with best game


        if done and game >= num_initial_observation and not PLAY_GAME:
            if save_weights and game%20 == 0 and game >35:
                #Save model
                #print("Saving weights")
                Qmodel.save_weights(weigths_filename)
                action_predictor_model.save_weights(apWeights_filename)

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

            break


plt.plot(mstats)
plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    Qmodel.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
