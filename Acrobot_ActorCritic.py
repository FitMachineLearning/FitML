'''
Acrobot solution by Michel Aka
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Using Actor Critic

Note that I prefe the terms Action Predictor Network and Q/Reward Predictor network better

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


num_env_variables = 6
num_env_actions = 1
num_initial_observation =20
learning_rate =  0.003
apLearning_rate = 0.001
weigths_filename = "Acrobot-QL-v2-weights.h5"
apWeights_filename = "Acrobot_ap-QL-v2-weights.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.99
max_memory_len = 60000
starting_explore_prob = 0.3
training_epochs = 2
load_previous_weights = False
observe_and_train = True
save_weights = True
num_games_to_play = 200


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('Acrobot-v1')
env.reset()

#initialize training matrix with random states and actions
dataX = np.random.random(( 5,num_env_variables+num_env_actions ))
#Only one output for the total score / reward
dataY = np.random.random((5,1))

#initialize training matrix with random states and actions
apdataX = np.random.random(( 5,num_env_variables ))
apdataY = np.random.random((5,num_env_actions))

#nitialize the Reward predictor model
model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(128, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(dataY.shape[1]))

opt = optimizers.adam(lr=learning_rate)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(128, activation='tanh', input_dim=apdataX.shape[1]))
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
dataX = np.zeros(shape=(1,num_env_variables+num_env_actions))
dataY = np.zeros(shape=(1,1))

memoryX = np.zeros(shape=(1,num_env_variables+num_env_actions))
memoryY = np.zeros(shape=(1,1))

apmemoryX = np.zeros(shape=(1,num_env_variables))
apmemoryY = np.zeros(shape=(1,num_env_actions))

print("dataX shape", dataX.shape)
print("dataY shape", dataY.shape)

mstats = np.array([0])


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
        gameX = np.zeros(shape=(1,num_env_variables+num_env_actions))
        gameY = np.zeros(shape=(1,1))
        #Get the Q state
        qs = env.reset()
        #print("qs ", qs)
        if game < num_initial_observation:
            print("Observing game ", game)
        else:
            print("Learning & playing game ", game)
        for step in range (1500):

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

                    #print("taking random action",a, "at total_steps" , total_steps)
                    #print("prob ", prob, "explore_prob", explore_prob)

                else:
                    ##chose an action by estimating function-estimator remembered consequences of all possible actions
                    ## Bellman states that the best policy (i.e. action) is the one that maximizez expected rewards for future states
                    ## to caculate rewards we compute the reward a this state t + the discounted (b_discount) reward at all possible state t+1
                    ## all states t+1 are estimated by our function estimator (our Neural Network)

                    #Get Remembered optiomal policy
                    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)

                    '''
                    #Generate a set of num_env_action*10
                    possible_actions = np.zeros(shape=(num_env_actions*4,num_env_actions))
                    utility_possible_actions = np.zeros(shape=(num_env_actions*4))

                    for i in range(num_env_actions*4):
                        possible_actions[i] = SmartCrossEntropy(remembered_optimal_policy)
                        utility_possible_actions[i] = predictTotalRewards(qs,possible_actions[i])

                    #print("utility_possible_actions", utility_possible_actions)
                    #chose argmax action of estimated anticipated rewards
                    #print("utility_possible_actions ",utility_possible_actions)
                    #print("argmax of utitity", np.argmax(utility_possible_actions))
                    best_sce_i = np.argmax(utility_possible_actions)
                    '''

                    randaction = np.array([env.action_space.sample()])

                    #Compare R for SmartCrossEntropy action with remembered_optimal_policy and select the best
                    #if predictTotalRewards(qs,remembered_optimal_policy) > utility_possible_actions[best_sce_i]:
                    if predictTotalRewards(qs,remembered_optimal_policy) > predictTotalRewards(qs,randaction):
                        a = remembered_optimal_policy
                        #print(" | selecting remembered_optimal_policy ",a)
                    else:
                        a = randaction
                        #print(" - selecting generated optimal policy ",a)



            #print("state", qs)
            #print("Taking Action",a)
            if a[0] <0:
                a [0]= 0
            if a[0] > 2:
                a[0] = 2

            env.render()
            a = np.around(a)
            a = a.astype(int)
            qs_a = np.concatenate((qs,a), axis=0)

            #print("action",a," qs_a",qs_a)
            #get the target state and reward
            s,r,done,info = env.step(a[0])
            #record only the first x number of states
            if(r >= 0):
                print("reward ", r, "at step ", step)


            if step ==0:
                gameX[0] = qs_a
                gameY[0] = np.array([r])
                memoryX[0] = qs_a
                memoryY[0] = np.array([r])
                apmemoryX[0] = qs
                apmemoryY[0] = a

            gameX = np.vstack((gameX,qs_a))
            gameY = np.vstack((gameY,np.array([r])))
            apmemoryX = np.vstack((apmemoryX,qs))
            apmemoryY = np.vstack((apmemoryY,a))



            if done :
                #GAME ENDED
                if game == 0:
                    mstats[0] = step
                else:
                    mstats = np.append(mstats, step)



                #Calculate Q values from end to start of game
                for i in range(0,gameY.shape[0]):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                        gameY[(gameY.shape[0]-1)-i][0] = gameY[(gameY.shape[0]-1)-i][0]
                    else:
                        #print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                        gameY[(gameY.shape[0]-1)-i][0] = gameY[(gameY.shape[0]-1)-i][0]+b_discount*gameY[(gameY.shape[0]-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])
                    if i==gameY.shape[0]-1:
                        print("Training Game #",game, " steps = ", step ,"last reward", r," finished with headscore ", gameY[(gameY.shape[0]-1)-i][0])

                if memoryX.shape[0] ==1:
                    memoryX = gameX
                    memoryY = gameY
                else:
                    #Add experience to memory
                    memoryX = np.concatenate((memoryX,gameX),axis=0)
                    memoryY = np.concatenate((memoryY,gameY),axis=0)

                #if memory is full remove first element
                if np.alen(memoryX) >= max_memory_len:
                    #print("memory full. mem len ", np.alen(memoryX))
                    for l in range(np.alen(gameX)):
                        memoryX = np.delete(memoryX, 0, axis=0)
                        memoryY = np.delete(memoryY, 0, axis=0)
                        apmemoryX = np.delete(apmemoryX, 0 , axis=0)
                        apmemoryY = np.delete(apmemoryY, 0 , axis=0)


            #Update the states
            qs=s

            #Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game%1 == 0:
                    print("Training  game# ", game,"momory size", memoryX.shape[0])

                    #training Reward predictor model
                    model.fit(memoryX,memoryY, batch_size=32,epochs=training_epochs,verbose=2)

                    #training action predictor model
                    action_predictor_model.fit(apmemoryX,apmemoryY, batch_size=32, epochs=training_epochs,verbose=2)

            if done and game >= num_initial_observation:
                if save_weights and game%20 == 0:
                    #Save model
                    print("Saving weights")
                    model.save_weights(weigths_filename)
                    action_predictor_model.save_weights(apWeights_filename)

            if done:
                #Game won  conditions
                if step > 197:
                    print("Game ", game," WON ***  steps",step  )
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
