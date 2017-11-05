'''
Cartpole-v0 solution by Michel Aka

https://github.com/FitMachineLearning/FitML/

Using DeepQ Learning

'''
import numpy as np
import keras
import gym
import os
import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers


num_env_variables = 4
num_env_actions = 2
num_training_exmaples = 30
timesteps = 1
num_initial_observation = 60
learning_rate = 0.001
weigths_filename = "Cartpole-weights_DQN_Mem_1Loop.h5"

b_discount = 0.95
max_memory_len = 2000
num_failures_for_retrain = 10
starting_explore_prob = 0.05
initial_training_epochs = 2
RL_training_eporcs = 2
num_anticipation_steps = 6
load_previous_weights = False
observe_and_train = True
Do_RL = True
save_weights = True
Learning_cycles = 1500


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('CartPole-v0')
env.reset()

#initialize training matrix with random states and actions
dataX = np.random.random(( num_training_exmaples,num_env_variables+num_env_actions ))
#Only one output for the total score
dataY = np.random.random((num_training_exmaples,1))



#nitialize the LSTM with random weights

model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(512, activation='relu', input_dim=dataX.shape[1]))
model.add(Dense(dataY.shape[1]))

opt = optimizers.adam(lr=learning_rate)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

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

#Record first 500 in a sequence and add them to the training sequence
total_steps = 0
dataX = np.zeros(shape=(1,num_env_variables+num_env_actions))
dataY = np.zeros(shape=(1,1))

memoryX = np.zeros(shape=(1,num_env_variables+num_env_actions))
memoryY = np.zeros(shape=(1,1))


print("dataX shape", dataX.shape)
print("dataY shape", dataY.shape)



def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,actions_1_hot[action]), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = model.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward



if observe_and_train:
    #observe for 100 games


    for game in range(500):
        gameX = np.zeros(shape=(1,num_env_variables+num_env_actions))
        gameY = np.zeros(shape=(1,1))
        #Get the Q state
        qs = env.reset()
        for step in range (1000):

            if game < num_initial_observation:
                #take a radmon action
                a = env.action_space.sample()
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/Learning_cycles)*game

                #Chose between prediction and chance
                if prob < explore_prob:
                    #take a random action
                    a=env.action_space.sample()
                    #print("taking random action",a, "at total_steps" , total_steps)
                    #print("prob ", prob, "explore_prob", explore_prob)

                else:
                    ##chose an action by estimating consequences of actions for the next num_anticipation_steps steps ahead
                    #works best with looking 6 steps ahead
                    #Also works best if you train the model more itterations
                    utility_possible_actions = np.zeros(shape=(num_env_actions))

                    utility_possible_actions[0] = predictTotalRewards(qs,0)
                    utility_possible_actions[1] = predictTotalRewards(qs,1)


                    #chose argmax action of estimated anticipated rewards
                    #print("utility_possible_actions ",utility_possible_actions)
                    #print("argmax of utitity", np.argmax(utility_possible_actions))
                    a = np.argmax(utility_possible_actions)



            env.render()
            qs_a = np.concatenate((qs,actions_1_hot[a]), axis=0)

            #print("action",a," qs_a",qs_a)
            #get the target state and reward
            s,r,done,info = env.step(a)
            #record only the first x number of states

            if done and step <=196:
                r = -1

            if step ==0:
                gameX[0] = qs_a
                gameY[0] = np.array([r])
                memoryX[0] = qs_a
                memoryY[0] = np.array([r])

            gameX = np.vstack((gameX,qs_a))
            gameY = np.vstack((gameY,np.array([r])))


            if done :
                #GAME ENDED
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
                    print("memory full. mem len ", np.alen(memoryX))
                    memoryX = np.delete(memoryX, 0, axis=0)
                    memoryY = np.delete(memoryY, 0, axis=0)

            #Update the states
            qs=s

            #Retrain every X failures after num_initial_observation
            if done and game >= num_env_actions:
                if game%10 == 0:
                    print("Training  game# ", game,"momory size", memoryX.shape[0])
                    model.fit(memoryX,memoryY, batch_size=32,epochs=initial_training_epochs,verbose=2)

            if done:
                #Game ended - Break
                break

    print("Observation complete. - Begin LSTM training")

    print("dataX shape", dataX.shape)
    print(dataX[0:20])
    print("dataY shape", dataY.shape)
    print(dataY)





    #The more epochs you train the model, the better is becomes at predicting future states
    #This in turn will improve the results of the Bellman equation and thus will lead us to
    # better decisions in our MDP process
    #model.fit(feedX,feedY, batch_size=32,epochs=initial_training_epochs,verbose=2)

    #print("total_steps ", total_steps)
    #print("dataX ", dataX[0:10])
    #print("dataY ", dataY[0:10])
    #print("dataY ", dataY)


#dataX = np.random.random((1,9))


if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
