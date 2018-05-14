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
from ple.games.flappybird import FlappyBird
from ple.games.pong import Pong
from ple.games.pixelcopter import Pixelcopter
from ple import PLE


import pygal
import os
import h5py
#import matplotlib.pyplot as plt
import math
import matplotlib

from random import gauss
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D,Flatten,Convolution2D
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers

from matplotlib import pyplot as plt
import skimage
from PIL import Image
from skimage import color,transform,exposure
from scipy.misc import toimage


PLAY_GAME = False #Set to True if you want to agent to play without training
uses_critic = True
uses_parameter_noising = True


IMG_DIM = 80

ENVIRONMENT_NAME = "Pong-v0"
game = Pixelcopter(width=160, height=160)
p = PLE(game, fps=30, display_screen=True)
num_env_variables = 8
num_env_actions = 2


num_initial_observation = 10
learning_rate =  0.0001
apLearning_rate = 0.01

MUTATION_PROB = 0.1

littl_sigma = 0.00006
big_sigma = 0.1
upper_delta = 0.0375
lower_delta = 0.015
#gaussSigma = 0.01
version_name = ENVIRONMENT_NAME + "ker_v11"
weigths_filename = version_name+"-weights.h5"
apWeights_filename = version_name+"-weights-ap.h5"


#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.98
max_memory_len = 30000
experience_replay_size = 512
random_every_n = 800
num_retries = 60
starting_explore_prob = 0.05
training_epochs = 3
mini_batch = 512
load_previous_weights = False
observe_and_train = True
save_weights = True
save_memory_arrays = True
load_memory_arrays = False
do_training = True
num_games_to_play = 200000
random_num_games_to_play = num_games_to_play/3
USE_GAUSSIAN_NOISE = True
CLIP_ACTION = True
HAS_REWARD_SCALLING = False
USE_ADAPTIVE_NOISE = True
HAS_EARLY_TERMINATION_REWARD = False
EARLY_TERMINATION_REWARD = -5
USES_CATEGORICAL_ACTIONS = False
USES_SELECTIVE_MEMORY = False
max_steps = 70400



#Selective memory settings
sm_normalizer = 20
sm_memory_size = 10500

last_game_average = -1000
last_best_noisy_game = -1000
max_game_average = -1000
num_positive_avg_games = 0

imgbuff0 = np.zeros(shape=(IMG_DIM,IMG_DIM))
imgbuff1 = np.zeros(shape=(IMG_DIM,IMG_DIM))
imgbuff2 = np.zeros(shape=(IMG_DIM,IMG_DIM))

#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment



#initialize training matrix with random states and actions
dataX = np.random.random(( 5,num_env_variables+num_env_actions )) #Irrelevant, set to input shape = channel*height*width
#Only one output for the total score / reward
dataY = np.random.random((5,num_env_actions))

#initialize training matrix with random states and actions
apdataX = np.random.random(( 5,num_env_variables ))
apdataY = np.random.random((5,num_env_actions))

def custom_error(y_true, y_pred, Qsa):
    cce=0.001*(y_true - y_pred)*Qsa
    return cce


Qmodel = Sequential()
Qmodel.add(Conv2D(32, (3, 3), activation='relu', subsample=(4, 4), input_shape=(1,IMG_DIM,IMG_DIM*3)))
Qmodel.add(Conv2D(64, (4, 4), activation='relu', subsample=(2, 2)))
Qmodel.add(Conv2D(64, (3, 3), activation='relu', name="dense_one" ))
Qmodel.add(Flatten())
Qmodel.add(Dense(512,activation='relu'))
#Qmodel.add(Dropout(0.3))


Qmodel.add(Dense(dataY.shape[1]))
opt = optimizers.adam(lr=learning_rate)
#opt = optimizers.Adadelta()

Qmodel.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


noisyModel = Sequential()
noisyModel.add(Conv2D(32, (3, 3), activation='relu', subsample=(4, 4), input_shape=(1,IMG_DIM,IMG_DIM*3)))
noisyModel.add(Conv2D(64, (4, 4), activation='relu', subsample=(2, 2)))
noisyModel.add(Conv2D(64, (3, 3), activation='relu', name="dense_one" ))
noisyModel.add(Flatten())
noisyModel.add(Dense(512,activation='relu'))


noisyModel.add(Dense(dataY.shape[1]))
opt2 = optimizers.adam(lr=learning_rate)
#opt = optimizers.Adadelta()

noisyModel.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])




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



memorySA = []
memoryA = []
memoryR = []


if load_memory_arrays:
    if os.path.isfile(version_name+'memorySA.npy'):
        print("Memory Files exist. Loading...")
        memorySA = np.load(version_name+'memorySA.npy')
        memoryA = np.load(version_name+'memoryA.npy')
        memoryR = np.load(version_name+'memoryR.npy')

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

#------ Parameter Noising
def add_gaussian_noise(mu,noiseSigma,largeNoise=False):
    #print ( gauss(mu, noiseSigma) )
    if np.random.rand(1) < MUTATION_PROB:
        return gauss(mu, noiseSigma)
    else:
        return mu+0.0

def add_noise_simple(mu, largeNoise=False):
    x =   np.random.rand(1) - 0.5 #probability of doing x
    if not largeNoise:
        x = x*big_sigma
    else:
        x = x*big_sigma   #Sigma = width of the standard deviaion
    #print ("x/200",x,"big_sigma",big_sigma)
    return mu + x


add_gaussian_noise = np.vectorize(add_gaussian_noise,otypes=[np.float])
add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])


def add_noise_to_model(noisyModel,largeNoise = False):
    #noisy_model = keras.models.clone_model(action_predictor_model)
    #noisy_model.set_weights(action_predictor_model.get_weights())
    #print("Adding Noise to actor")
    #largeNoise =  last_game_average < memoryR.mean()
    sz = len(noisyModel.layers)
    #if largeNoise:
    #    print("Setting Large Noise!")
    for k in range(sz):
        w = noisyModel.layers[k].get_weights()
        if np.alen(w) >0 and noisyModel.layers[k].name == 'dense_3':
            print("Layer Name", noisyModel.layers[k].name,w[0].shape)

            #print("k==>",k)
            if USE_GAUSSIAN_NOISE:
                w[0] = add_gaussian_noise(w[0],big_sigma,largeNoise)
            else:
                w[0] = add_noise_simple(w[0],largeNoise)

        noisyModel.layers[k].set_weights(w)
    return noisyModel

def reset_noisy_model_weights_to_apWeights(mu):
    x =  mu+0.0 #probability of doing x
    return x

reset_noisy_model_weights_to_apWeights = np.vectorize(reset_noisy_model_weights_to_apWeights,otypes=[np.float])

def reset_noisy_model(targetModel,mainActorModel):
    sz = len(targetModel.layers)
    #if largeNoise:
    #    print("Setting Large Noise!")
    for k in range(sz):
        w = targetModel.layers[k].get_weights()
        apW = mainActorModel.layers[k].get_weights()

        if np.alen(w) >0:
            w[0] = reset_noisy_model_weights_to_apWeights(apW[0])
            #w[0] = apW[0]+0.0
        targetModel.layers[k].set_weights(w)
        #print("w",w)
        #print("apW",apW)
    return targetModel

#-----

#takes a single game frame as input
#preprocesses before feeding into model

def preprocessing2(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """

  I = I[35:195] # crop

  #print("x_t1",I.shape)
  x_t1 = skimage.color.rgb2gray(I)
  x_t1 = skimage.transform.resize(x_t1,(IMG_DIM,IMG_DIM))
  x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
  #print("x_t1",x_t1.shape)
  #print("x_t1",x_t1)

  #x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
  #s_t1 = np.append(x_t1, s_t1[:, :3, :, :], axis=1)

  return x_t1 #flattens

def appendBufferImages(img1,img2,img3):

    new_im = np.concatenate((img1,img2,img3),axis=1)

    #matplotlib.pyplot.imshow(new_im)
    #matplotlib.pyplot.show()
    return new_im



# --- Parameter Noising


def DeepQPredictBestAction(qstate,is_noisy_game = False):
    qs_a = qstate
    predX = np.zeros(shape=(1,IMG_DIM,IMG_DIM*3))
    predX[0] = qs_a
    #print("qs_a",qs_a.shape)
    #img = predX[0].reshape(1,1,IMG_DIM,IMG_DIM*3)
    #toimage(img[0][0]).show()
    #print("trying to predict reward at qs_a", predX[0])
    if is_noisy_game:
        pred = Qmodel.predict(predX[0].reshape(1,1,IMG_DIM,IMG_DIM*3))
    else:
        pred = noisyModel.predict(predX[0].reshape(1,1,IMG_DIM,IMG_DIM*3))

    remembered_total_reward = pred[0]
    return remembered_total_reward


p.act(118)
p.game_over()
p.getScreenRGB()

p.act(118)
p.game_over()
p.getScreenRGB()
p.act(118)
p.game_over()
p.getScreenRGB()
p.act(118)
p.game_over()
p.getScreenRGB()
p.act(119)
p.game_over()
p.getScreenRGB()

#Play the game 500 times
for game in range(num_games_to_play):
    #gameSA = np.zeros(shape=(1,IMG_DIM,IMG_DIM*3))
    #gameA = np.zeros(shape=(1,num_env_actions))
    #gameR = np.zeros(shape=(1,1))
    #gameI  = np.zeros(shape=(1,1))

    gameSA = []
    gameA = []
    gameR = []
    gameI  = []

    #print("gameSA",gameSA)

    p.reset_game()
    qs = p.getScreenRGB()

    imgbuff0 = preprocessing2(qs)
    imgbuff1 = preprocessing2(qs)
    imgbuff2 = preprocessing2(qs)


    mAP_Counts = 0
    num_add_mem = 0
    #print("qs ", qs)
    is_noisy_game = False
    last_prediction = np.zeros(num_env_actions)

    #noisy_model.set_weights(action_predictor_model.get_weights())

    #Add noise to Actor
    if game > num_initial_observation and uses_parameter_noising:
        is_noisy_game = False
        #print("Adding Noise")
        if (game%2==0 ):
            is_noisy_game = True
            if game%20==0:
                #print("Adding BIG Noise")
                #noisy_model = keras.models.clone_model(action_predictor_model)
                print("Reseting Model ...")
                noisyModel = reset_noisy_model(noisyModel,Qmodel)
                print("Adding Noise ...")
                noisy_model = add_noise_to_model(noisyModel,True)
                #last_best_noisy_game = -1000
                print("Done Param Noise")


    for step in range (5000):


        imgbuff2 = imgbuff1
        imgbuff1 = imgbuff0
        imgbuff0 = preprocessing2(qs)

        qs = appendBufferImages(imgbuff0,imgbuff1,imgbuff2)
        #if step%300==1:
        #    toimage(qs).show()
        index = 0

        #if PLAY_GAME:
        #    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)
        #    a = remembered_optimal_policy
        #print("last_prediction",last_prediction)
        if game < num_initial_observation:
            #take a radmon action
            a = np.argmax (keras.utils.to_categorical(np.random.randint(2),num_env_actions) )
        else:
            prob = np.random.rand(1)
            explore_prob = starting_explore_prob-(starting_explore_prob/random_num_games_to_play)*game

            if game > random_num_games_to_play:
                prob = 0.000001
            #Chose between prediction and chance
            if prob < explore_prob or game%random_every_n==1:
                #take a random action
                last_prediction = DeepQPredictBestAction(qs,is_noisy_game)

                a = np.argmax (keras.utils.to_categorical(np.random.randint(2),num_env_actions) )
                index = a
            else:
                last_prediction = DeepQPredictBestAction(qs,is_noisy_game)
                if step%50==1:
                    print("prediction from actor ",last_prediction)

                a = np.argmax(last_prediction)
                index = a


        #print("a",a)
        #env.render()
        qs_a = qs

        #get the target state and reward
        r = p.act(a+118)
        done = p.game_over()
        s = p.getScreenRGB()
        #record only the first x number of states

        #print("r",r)
        #if r<0:
        #    print("negative reward")
        #if r>0:
        #    print("++")



        if HAS_REWARD_SCALLING:
            r=r/200 #reward scalling to from [-1,1] to [-100,100]

        if USES_CATEGORICAL_ACTIONS:
            a = keras.utils.to_categorical(a,num_env_actions)
            #set action array index to reward

        else:
            last_prediction[a] = r

            a = last_prediction
            #if step%50==1:
            #    print("a",a)


        gameSA.append( qs_a.reshape(1,IMG_DIM,IMG_DIM*3))
        #if step%300 == 1:
        #    toimage(gameSA[step-2][0]).show()
        gameR.append( [r])
        gameA.append( a)
        gameI.append([index])

        if step > max_steps:
            done = True

        if done :
            tempGameSA = []
            tempGameA = []
            tempGameR = []

            #Calculate Q values from end to start of game
            #mstats.append(step)
            for i in range(0,len(gameR)):
                #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                if i==0:
                    #print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                    gameR[(len(gameR)-1)-i][0] = gameR[(len(gameR)-1)-i][0]
                else:
                    #print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                    gameR[(len(gameR)-1)-i][0] = gameR[(len(gameR)-1)-i][0]+b_discount*gameR[(len(gameR)-1)-i+1][0]
                    #print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])

            for i in range(np.alen(gameR)):
                action = gameA[i]
                indx = gameI[i][0]
                action[indx] = gameR[i][0]
                gameA[i] = action
                #print("gameA[i]",gameA[i])


            memorySA = memorySA+ gameSA
            memoryR = memoryR+ gameR
            memoryA = memoryA+ gameA

            if np.mean(gameR) > max_game_average :
                max_game_average = np.mean(gameR)

            #if memory is full remove first element
            if np.alen(memoryR) >= max_memory_len:
                memorySA = memorySA[len(gameR):]
                memoryR = memoryR[len(gameR):]
                memoryA = memoryA[len(gameR):]


        qs=s

        if done and game > num_initial_observation and not PLAY_GAME:
            last_game_average = np.mean(gameR)
            if is_noisy_game and last_game_average > np.mean(memoryR):
                last_best_noisy_game = last_game_average
            #if game >3:
                #actor_experience_replay(gameSA,gameR,gameS,gameA,gameW,1)

            #if game > 3 and game %1 ==0:
                # train on all memory
                #for i in range(3):

                #actor_experience_replay(memorySA,memoryR,memoryS,memoryA,memoryW,training_epochs)
            if game > 1 and game %10 ==0 and uses_critic:
                for t in range(training_epochs):
                    print("Experience Replay")
                    tSA = np.asarray(memorySA)
                    tA = np.asarray(memoryA)
                    tR = np.asarray(memoryR)

                    if USES_SELECTIVE_MEMORY:
                        stdDev = np.std(tR)
                        treshold = tR.mean() + stdDev
                        train_C = np.arange(np.alen(tR))
                        train_C = train_C[tR.flatten()> treshold] # Only take games that are above gameTreshold
                        tSA = tSA[train_C,:]
                        tA = tA[train_C,:]
                        tR = tR[train_C,:]
                        print("Selected after treshold ", np.alen(tR))

                    train_A = np.random.randint(tR.shape[0],size=int(min(experience_replay_size,np.alen(tA) )))
                    num_records = np.alen(train_A)
                    tA = tA[train_A,:]
                    tSA = tSA[train_A,:]
                    tR = tR[train_A,:]
                    #print("Training Critic n elements =", np.alen(tR),"treshold",treshold)
                    tSA = tSA.reshape(num_records,1,IMG_DIM,IMG_DIM*3)
                    #toimage(tSA[0][0]).show()
                    Qmodel.fit(tSA ,tA, batch_size=mini_batch, nb_epoch=1,verbose=0)



        if done and game >= num_initial_observation and not PLAY_GAME:
            if save_weights and game%5 == 0 and game >35:
                #Save model
                #print("Saving weights")
                Qmodel.save_weights(weigths_filename)
                #action_predictor_model.save_weights(apWeights_filename)

            if save_memory_arrays and game%20 == 0 and game >35:
                np.save(version_name+'memorySA.npy',np.asarray(memorySA))
                np.save(version_name+'memoryA.npy',np.asarray(memoryA))
                np.save(version_name+'memoryR.npy',np.asarray(memoryR))




        if done:
            oldAPCount = mAP_Counts
            if np.mean(gameR) >0:
                num_positive_avg_games += 1
            if game%1==0:
                #print("Training Game #",game,"last everage",memoryR.mean(),"max_game_average",max_game_average,,"game mean",gameR.mean(),"memMax",memoryR.max(),"memoryR",memoryR.shape[0], "SelectiveMem Size ",memoryRR.shape[0],"Selective Mem mean",memoryRR.mean(axis=0)[0], " steps = ", step )
                if is_noisy_game:
                    print("Noisy Game #  %7d  avgScore %8.3f  last_game_avg %8.3f  max_game_avg %8.3f  memory size %8d memMax %8.3f steps %5d pos games %5d" % (game, np.mean(memoryR), last_game_average, np.mean(memoryR) , len(memoryR), np.max(memoryR), step,num_positive_avg_games    ) )
                else:
                    print("Reg Game   #  %7d  avgScore %8.3f  last_game_avg %8.3f  max_game_avg %8.3f  memory size %8d memMax %8.3f steps %5d pos games %5d" % (game, np.mean(memoryR), last_game_average, np.mean(memoryR) , len(memoryR), np.max(memoryR), step,num_positive_avg_games    ) )


            break



if save_weights:
    #Save model
    print("Saving weights")
    Qmodel.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
