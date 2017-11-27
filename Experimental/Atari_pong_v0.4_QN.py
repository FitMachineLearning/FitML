'''
Atari Pong solution by Michel Aka
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Using DeepQ N
'''
import numpy as np
import keras
import gym
import os
import h5py

import matplotlib
from matplotlib import pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


num_env_variables = 80*80*2 # each sate is 2 frames of the pong game
num_env_actions = 1
num_initial_observation = 30
learning_rate =  0.001
apLearning_rate = 0.002
weigths_filename = "Pong-QL-v0-weights.h5"
apWeights_filename = "Pong-QL-v0-weights-ap.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.99
max_memory_len = 10000
starting_explore_prob = 0.05
training_epochs = 3
mini_batch = 128
load_previous_weights = True
observe_and_train = True
save_weights = True
num_games_to_play = 3000


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('Pong-v0')
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
model.add(Dense(2048, activation='relu', input_dim=dataX.shape[1]))
model.add(Dropout(0.25))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(32, activation='tanh'))
#model.add(Dropout(0.25))
model.add(Dense(dataY.shape[1]))

opt = optimizers.adam(lr=learning_rate)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(256, activation='relu', input_dim=apdataX.shape[1]))
action_predictor_model.add(Dense(32, activation='relu'))
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
num_games_won = 0


#takes a single game frame as input
#preprocesses before feeding into model
def preprocessing(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
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
        gameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
        gameS = np.zeros(shape=(1,num_env_variables))
        gameA = np.zeros(shape=(1,1))
        gameR = np.zeros(shape=(1,1))

        previous_state = np.zeros(80*80)
        #Get the Q state
        qs = env.reset()

        #print("qs ", qs)
        for step in range (7000):

            qs = preprocessing(qs)
            #if np.array_equal(qs,previous_state):
                #print("Previous state = to Qstate we have a problem")
            sequenceQS = np.concatenate((previous_state,qs),axis=0)

            #if step%10==0:
                #plt.imshow(np.reshape(sequenceQS,(-1,80)))
                #plt.show()

            if game < num_initial_observation or game%10==0:
                #take a radmon action
                a = np.array([env.action_space.sample()])
                #if 'info' in locals():
                #    print(step, "random action ",a,"reward",r, "info", info)
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob:
                    #take a random action
                    a=np.array([env.action_space.sample()])

                else:

                    #Get Remembered optiomal policy
                    #remembered_optimal_policy = GetRememberedOptimalPolicy(qs)

                    #if step %25 == 0:
                    #    print("remembered_optimal_policy", remembered_optimal_policy)

                    #randaction = np.array([env.action_space.sample()])

                    #Compare R for SmartCrossEntropy action with remembered_optimal_policy and select the best
                    #if predictTotalRewards(qs,remembered_optimal_policy) > utility_possible_actions[best_sce_i]:



                    predictedRewards = np.zeros(6)
                    for i in range(6):
                        predictedRewards[i] = predictTotalRewards(sequenceQS,np.array([i]))

                    #print("predictedRewards",predictedRewards)
                    a = np.argmax(predictedRewards)


                    a = np.array([a])


            if a[0] <0:
                a [0]= 0
            if a[0] > 5:
                a[0] = 5

            env.render()
            a = np.around(a)
            a = a.astype(int)
            qs_a = np.concatenate((sequenceQS,a), axis=0)


            #get the target state and reward
            s,r,done,info = env.step(a[0])
            #record only the first x number of states

            if r>0:
                print("positive reward ",r)
            if step ==0:
                gameSA[0] = qs_a
                #gameS[0] = sequenceQS
                gameR[0] = np.array([r])
                #gameA[0] = np.array([a])
            else:
                gameSA= np.vstack((gameSA, qs_a))
                #gameS= np.vstack((gameS, sequenceQS))
                gameR = np.vstack((gameR, np.array([r])))
                #gameA = np.vstack((gameA, np.array([a])))

            if step > 1498:
                done = True


            if done or r == 1 or r == -1:
                done = True
                if r ==1:
                    num_games_won +=1
                tempGameSA = np.zeros(shape=(1,num_env_variables+num_env_actions))
                #tempGameS = np.zeros(shape=(1,num_env_variables))
                #tempGameA = np.zeros(shape=(1,num_env_actions))
                tempGameR = np.zeros(shape=(1,1))
                #tempGameRR = np.zeros(shape=(1,1))

                #Calculate Q values from end to start of game
                mstats.append(step)
                for i in range(0,gameR.shape[0]):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameR[(gameR.shape[0]-1)-i][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]
                    else:
                        #print("local error before Bellman", gameR[(gameR.shape[0]-1)-i][0],"Next error ", gameR[(gameR.shape[0]-1)-i+1][0])
                        gameR[(gameR.shape[0]-1)-i][0] = gameR[(gameR.shape[0]-1)-i][0]+b_discount*gameR[(gameR.shape[0]-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameR[(gameR.shape[0]-1)-i][0])
                    if i==gameR.shape[0]-1:
                        print("Training Game #",game, "# wins ", num_games_won,"avg win",num_games_won/(game+1), "memory ",memoryR.shape[0]," finished with headscore ", gameR[(gameR.shape[0]-1)-i][0])

                if memoryR.shape[0] ==1:
                    memorySA = gameSA
                    memoryR = gameR
                    #memoryA = gameA
                    #memoryS = gameS
                #else:

                #tempGameA = tempGameA[1:]
                #tempGameS = tempGameS[1:]
                #tempGameRR = tempGameRR[1:]
                tempGameR = tempGameR[1:]
                tempGameSA = tempGameSA[1:]

                for i in range(gameR.shape[0]):
                    if np.random.rand(1) < 0.06:
                        tempGameSA = np.vstack((tempGameSA,gameSA[i]))
                        tempGameR = np.vstack((tempGameR,gameR[i]))
                '''
                    #Add experience to memory
                    memorySA = np.concatenate((memorySA,gameSA),axis=0)
                    #memoryS = np.concatenate((memoryS,gameS),axis=0)
                    memoryR = np.concatenate((memoryR,gameR),axis=0)
                    #memoryA = np.concatenate((memoryA,gameA),axis=0)
                '''
                if memoryR.shape[0] ==1:
                    memoryR = tempGameR
                    memorySA = tempGameSA
                else:
                    memorySA = np.concatenate((memorySA,tempGameSA),axis=0)
                    memoryR = np.concatenate((memoryR,tempGameR),axis=0)



                #if memory is full remove first element
                if np.alen(memoryR) >= max_memory_len:
                    memoryR = memoryR[np.alen(gameR):]
                    memorySA = memorySA[np.alen(gameR):]
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
                if game%3 == 0:
                    print("Training  game# ", game,"momory size", memorySA.shape[0])

                    #training Reward predictor model
                    model.fit(memorySA,memoryR, batch_size=mini_batch,epochs=training_epochs,verbose=0)

                    #training action predictor model
                    #action_predictor_model.fit(memoryS,memoryA, batch_size=mini_batch, epochs=training_epochs,verbose=0)

            if done and game >= num_initial_observation:
                if save_weights and game%20 == 0:
                    #Save model
                    print("Saving weights")
                    model.save_weights(weigths_filename)
                    action_predictor_model.save_weights(apWeights_filename)

            if done:
                #Game won  conditions
                if r > 0:
                    print("Game ", game," WON ***, info", info )
                else:
                    print("Game ",game," ended with reward.",r," info", info)
                #Game ended - Break
                break


plt.plot(mstats)
plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
