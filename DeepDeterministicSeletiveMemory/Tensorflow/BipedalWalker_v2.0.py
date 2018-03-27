'''
Mujoco HalfCheetah Walker with
 - Selective Memory
 - Actor Critic
 - Parameter Noising
 - Q as discriminator
solution by Michel Aka author of FitML github blog and repository
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Update
Deep Network
Starts to crawl at 78

Adagrad
0.99 delta
0.1 dropout

'''
import tensorflow as tf
import numpy as np
import gym
#import pybullet
#import pybullet_envs

import pygal
import os
import h5py
#import matplotlib.pyplot as plt
import math

from keras import optimizers


PLAY_GAME = False #Set to True if you want to agent to play without training
uses_critic = True
uses_parameter_noising = False

num_env_variables = 24
num_env_actions = 4
num_initial_observation = 6
learning_rate =  0.003
apLearning_rate = 0.001
ENVIRONMENT_NAME = "BipedalWalker-v2"
version_name = ENVIRONMENT_NAME + "With_PN_v7"
weigths_filename = version_name+"-weights.h5"
apWeights_filename = version_name+"-weights-ap.h5"


#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.99
max_memory_len = 90000
experience_replay_size = 45000
random_every_n = 50
num_retries = 30
starting_explore_prob = 0.05
training_epochs = 100
mini_batch = 512
load_previous_weights = False
observe_and_train = True
save_weights = True
save_memory_arrays = True
load_memory_arrays = False
do_training = True
num_games_to_play = 20000
random_num_games_to_play = num_games_to_play/3
max_steps = 840

#Selective memory settings
sm_normalizer = 20
sm_memory_size = 10500

last_game_average = -1000
last_best_noisy_game = -1000
max_game_average = -1000


#Create testing enviroment

env = gym.make(ENVIRONMENT_NAME)
env.render(mode="human")
env.reset()


print("-- Observations",env.observation_space)
print("-- actionspace",env.action_space)

#initialize training matrix with random states and actions
#dataX = np.random.random(( 5,num_env_variables+num_env_actions ))
dataX = tf.placeholder("float", [None, num_env_variables+num_env_actions])

#Only one output for the total score / reward
#dataY = np.random.random((5,1))
dataY = tf.placeholder("float", [None, 1])


#initialize training matrix with random states and actions
#apdataX = np.random.random(( 5,num_env_variables ))
apdataX = tf.placeholder("float", [None, num_env_variables])
#apdataY = np.random.random((5,num_env_actions))
apdataY = tf.placeholder("float", [None, num_env_actions])


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def Qmodel(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def apModel(X, apw_h, apw_o):
    h = tf.nn.relu(tf.matmul(X, apw_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, apw_o) # note that we dont take the softmax at the end because our cost fn does that for us

''' QModel '''
Qw_h = init_weights([num_env_variables+num_env_actions, 1024]) # create symbolic variables
Qw_o = init_weights([1024, 1])

Qpy_x = Qmodel(dataX, Qw_h, Qw_o)


Qcost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(dataY, Qpy_x))))
Qoptimizer = tf.train.AdadeltaOptimizer(1.,0.9,1e-6)
Qtrain_op = Qoptimizer.minimize(Qcost)

''' apModel '''
apw_h = init_weights([num_env_variables, 1024]) # create symbolic variables
apw_o = init_weights([1024, num_env_actions])

appy_x = apModel(apdataX, apw_h, apw_o)

apcost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(apdataY, appy_x))))
apOptimizer = tf.train.AdadeltaOptimizer(1.,0.9,1e-6)
aptrain_op = apOptimizer.minimize(apcost)




''' AP & NOISY MODEL
#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(64, activation='relu', input_dim=apdataX.shape[1]))
#action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(64, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(64, activation='relu'))
#action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(64, activation='relu'))
#action_predictor_model.add(Dropout(0.5))

action_predictor_model.add(Dense(apdataY.shape[1]))
#opt2 = optimizers.adam(lr=apLearning_rate)
opt2 = optimizers.Adadelta()

action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])


#initialize the action predictor model
noisy_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
noisy_model.add(Dense(2048, activation='relu', input_dim=apdataX.shape[1]))
#noisy_model.add(Dropout(0.5))
#noisy_model.add(Dense(256, activation='relu'))
#noisy_model.add(Dropout(0.5))
noisy_model.add(Dense(apdataY.shape[1]))
opt3 = optimizers.Adadelta()

noisy_model.compile(loss='mse', optimizer=opt3, metrics=['accuracy'])
'''

''' naModel '''
naw_h = init_weights([num_env_variables, 625]) # create symbolic variables
naw_o = init_weights([625, num_env_actions])

napy_x = apModel(apdataX, apw_h, apw_o)

nacost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(apdataY, napy_x))))
natrain_op = tf.train.AdadeltaOptimizer(1).minimize(nacost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


''' LOAD MODEL
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
'''

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
num_add_mem = 0
mAPPicks = []

# --- Parameter Noising
def add_noise(mu, largeNoise=False):

    if not largeNoise:
        sig = 0.00006
    else:
        #print("Adding Large parameter noise")
        sig = 0.006 #Sigma = width of the standard deviaion
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
        x = x/80
    else:
        x = x/10   #Sigma = width of the standard deviaion
    #print ("x/200",x)
    return mu + x


#add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])
add_noise = np.vectorize(add_noise,otypes=[np.float])
add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])


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
            w[0] = add_noise(w[0],largeNoise)

        targetModel.layers[k].set_weights(w)
    return targetModel




def reset_noisy_model_weights_to_apWeights(mu):
    x =  mu+(np.random.rand(1) - 0.5) / 10000000 #probability of doing x
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


# --- Parameter Noising

def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    #pred = Qmodel.predict(predX[0].reshape(1,predX.shape[1]))
    inputVal = predX[0].reshape(1,predX.shape[1])
    pred = sess.run(Qpy_x, feed_dict={dataX: inputVal})
    remembered_total_reward = pred[0][0]
    return remembered_total_reward

def GetRememberedOptimalPolicy(qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    #pred = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))

    inputVal = predX[0].reshape(1,predX.shape[1])
    pred = sess.run(appy_x, feed_dict={apdataX: inputVal})
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy

def GetRememberedOptimalPolicyFromNoisyModel(targetModel,qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    #pred = targetModel.predict(predX[0].reshape(1,predX.shape[1]))
    inputVal = predX[0].reshape(1,predX.shape[1])
    pred = sess.run(napy_x, feed_dict={apdataX: inputVal})

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


def pr_actor_experience_replay(memSA,memR,memS,memA,memW,num_epoch=1):
    tSA = (memSA)
    tR = (memR)
    tX = (memS)
    tY = (memA)
    tW = (memW)


    tX_train = np.zeros(shape=(1,num_env_variables))
    tY_train = np.zeros(shape=(1,num_env_actions))
    for i in range(np.alen(tR)):
        pr = predictTotalRewards(tX[i],GetRememberedOptimalPolicy(tX[i]))
        #print ("tR[i]",tR[i],"pr",pr)
        d = math.fabs( memoryR.max() - pr)
        tW[i]= 0.0000000000000005
        if (tR[i]>pr):
            tW[i]=0.15
        if (tR[i]>pr+d/2) or tR[i] > max_game_average:
            tW[i] = 1
        if tW[i]> np.random.rand(1):
            tX_train = np.vstack((tX_train,tX[i]))
            tY_train = np.vstack((tY_train,tY[i]))


    tX_train = tX_train[1:]
    tY_train = tY_train[1:]
    print("%8d were better After removing first element"%np.alen(tX_train))
    if np.alen(tX_train)>0:
        #action_predictor_model.fit(tX_train,tY_train, batch_size=mini_batch, nb_epoch=num_epochs,verbose=0)
        for num_epoch in range(training_epochs):
            sess.run(aptrain_op, feed_dict={apdataX: tX_train, apdataY: tY_train})





def actor_experience_replay(memSA,memR,memS,memA,memW,num_epoch=1):
    tSA = (memSA)
    tR = (memR)
    tX = (memS)
    tY = (memA)
    tW = (memW)

    target = tR.mean() #+ math.fabs( tR.mean() - tR.max()  )/2 #+ math.fabs( tR.mean() - tR.max()  )/4
    train_C = np.arange(np.alen(tR))
    train_C = train_C[tR.flatten()>target]
    tX = tX[train_C,:]
    tY = tY[train_C,:]
    tW = tW[train_C,:]
    tR = tR[train_C,:]

    train_A = np.random.randint(tY.shape[0],size=int(min(experience_replay_size,np.alen(tR) )))

    tX = tX[train_A,:]
    tY = tY[train_A,:]
    tW = tW[train_A,:]
    tR = tR[train_A,:]

    train_B = np.arange(np.alen(tR))

    tX_train = np.zeros(shape=(1,num_env_variables))
    tY_train = np.zeros(shape=(1,num_env_actions))
    for i in range(np.alen(train_B)):
        #pr = predictTotalRewards(tX[i],tY[i])
        ''' YOU CAN"T USE predictTotalRewards
        IF YOU DON"T TRAIN THE QMODEL

        if tR[i][0] < pr:
            tW[i][0] = -1
        else:
        '''
        d = math.fabs( memoryR.max() - target)
        tW[i] =  math.fabs(tR[i]-(target+0.000000000005)) / d
        #tW[i] = math.exp(1-(1/tW[i]**2))


        tW[i]= 0.0000000000000005
        if (tR[i]>target):
            tW[i]=0.5
        if (tR[i]>max_game_average):
            tW[i] = 1

        if tW[i]> np.random.rand(1):
            tX_train = np.vstack((tX_train,tX[i]))
            tY_train = np.vstack((tY_train,tY[i]))


            #print ("tW",tW[i],"exp", math.exp(1-(1/tW[i]**2)))
            #tW[i] = math.exp(1-(1/tW[i]**2))
            #tW[i] =  1
        #print("tW[i] %3.1f tR %3.2f target %3.2f max_game_average %3.2f "%(tW[i],tR[i],target,max_game_average))
    '''
    train_B = train_B[tW.flatten()>0]

    #print("%8d were better results than pr"%np.alen(tX_train))

    tX = tX[train_B,:]
    tY = tY[train_B,:]
    tW = tW[train_B,:]
    tR = tR[train_B,:]
    #print("tW",tW)
    '''
    #print("%8d were better results than pr"%np.alen(tX_train))
    ''' REMOVE FIRST ELEMENT BEFORE TRAINING '''
    tX_train = tX_train[1:]
    tY_train = tY_train[1:]
    #print("%8d were better After removing first element"%np.alen(tX_train))
    if np.alen(tX_train)>0:
        #tW = scale_weights(tR,tW)
        #print("# setps short listed ", np.alen(tR))

        #action_predictor_model.fit(tX_train,tY_train, batch_size=mini_batch, nb_epoch=num_epochs,verbose=0)
        for num_epoch in range(training_epochs):
            sess.run(aptrain_op, feed_dict={apdataX: tX_train, apdataY: tY_train})



def train_noisy_actor():
    tX = (memoryS)
    tY = (memoryA)
    tW = (memoryW)

    train_A = np.random.randint(tY.shape[0],size=int(min(experience_replay_size,np.alen(tY) )))
    tX = tX[train_A,:]
    tY = tY[train_A,:]
    tW = tW[train_A,:]

    #noisy_model.fit(tX,tY, batch_size=mini_batch, nb_epoch=training_epochs,verbose=0)
    for num_epoch in range(training_epochs):
        sess.run(natrain_op, feed_dict={apdataX: tX, apdataY: tY})



def add_controlled_noise(targetModel,largeNoise = False):
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

    while delta > 6 and deltaCount <20:
        #noisy_model.set_weights(action_predictor_model.get_weights())
        targetModel = add_noise_to_model(targetModel,largeNoise)
        for i in range(np.alen(tX)):
            a = GetRememberedOptimalPolicy(tX[i])
            b = GetRememberedOptimalPolicyFromNoisyModel(targetModel,tX[i])
            a = a.flatten()
            b = b.flatten()
            c = np.abs(a-b)
            diffs[i] = c.mean()
        delta = np.average (diffs)
        deltaCount+=1
    print("Tried x time ", deltaCount,"delta =", delta)
    return targetModel




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
    if game > num_initial_observation+4 and uses_parameter_noising:
        is_noisy_game = False
        #print("Adding Noise")
        if (game%2==0 ):
            is_noisy_game = True
            if last_best_noisy_game < memoryR.mean() or game%6==0:
                print("Adding BIG Noise")
                #noisy_model = keras.models.clone_model(action_predictor_model)
                reset_noisy_model()
                add_controlled_noise(noisy_model,True)
                #last_best_noisy_game = -1000
            else:
                print("Adding Small Noise")
                #print("Not Changing weights last_best_noisy_game", last_best_noisy_game," mean ",memoryR.mean())
                reset_noisy_model()
                add_controlled_noise(noisy_model,False)


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
                    remembered_optimal_policy = GetRememberedOptimalPolicyFromNoisyModel(action_predictor_model,qs)
                else:
                    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)
                a = remembered_optimal_policy

                if uses_critic:
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





        env.render()
        qs_a = np.concatenate((qs,a), axis=0)

        #get the target state and reward
        s,r,done,info = env.step(a)
        #record only the first x number of states

        if done and step<max_steps-3:
            r = -50

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


            for i in range(0,gameR.shape[0]):
                pr = predictTotalRewards(gameS[i],gameA[i])
                atm,add_prob = addToMemory(gameR[i][0], pr, max_game_average,memoryR.mean(),gameR.mean(axis=0)[0],np.std(memoryR))
                if add_prob < 0:
                    add_prob = 0.000000005
                #print("add_prob",add_prob)
                tempGameA = np.vstack((tempGameA,gameA[i]))
                tempGameS = np.vstack((tempGameS,gameS[i]))
                tempGameRR = np.vstack((tempGameRR,gameR[i]))
                tempGameW = np.vstack((tempGameW,add_prob))


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

            if game > 3 and game %2 ==0:
                # train on all memory
                print("Experience Replay")
                #for i in range(3):

                pr_actor_experience_replay(memorySA,memoryR,memoryS,memoryA,memoryW,training_epochs)
            if game > 3 and game %1 ==0 and uses_critic:
                tSA = (memorySA)
                tR = (memoryR)
                train_A = np.random.randint(tR.shape[0],size=int(min(experience_replay_size,np.alen(tR) )))
                tR = tR[train_A,:]
                tSA = tSA    [train_A,:]
                print("Training Critic n elements =", np.alen(tR))
                for num_epoch in range(training_epochs):
                    sess.run(Qtrain_op, feed_dict={dataX: tSA, dataY: tR})

                #Qmodel.fit(tSA,tR, batch_size=mini_batch, nb_epoch=training_epochs,verbose=0)
            if game > 3 and game %5 ==-1 and uses_parameter_noising:
                print("Training noisy_actor")
                train_noisy_actor()
                #Reinforce training with best game

        ''' SAVE MODEL
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

        '''


        if done:

            if game%1==0:
                #print("Training Game #",game,"last everage",memoryR.mean(),"max_game_average",max_game_average,,"game mean",gameR.mean(),"memMax",memoryR.max(),"memoryR",memoryR.shape[0], "SelectiveMem Size ",memoryRR.shape[0],"Selective Mem mean",memoryRR.mean(axis=0)[0], " steps = ", step )
                print(" #  %7d  avgScore %8.3f  last_game_avg %8.3f  max_game_avg %8.3f  memory size %8d memMax %8.3f steps %5d" % (game, memoryR.mean(), last_game_average, max_game_average , memoryR.shape[0], memoryR.max(), step    ) )

            if game%5 ==0 and np.alen(memoryR)>1000:
                mGames.append(game)
                mSteps.append(step/1000*100)
                mAPPicks.append(mAP_Counts/step*100)
                mAverageScores.append(max(memoryR.mean(), -50)/100*100)
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
