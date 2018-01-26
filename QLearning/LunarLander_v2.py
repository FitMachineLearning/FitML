'''
LunarLander-v2 solution by Michel Aka
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Using Modified Q Learning, Bellman, Reinforcement Learning, RL memory

'''
import numpy as np
import keras
import gym
import os
import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras import optimizers


num_env_variables = 8
num_env_actions = 4
num_initial_observation = 15
learning_rate = 0.003
weigths_filename = "LL-QL-v2-weights.h5"

b_discount = 0.99
max_memory_len = 60000
starting_explore_prob = 0.05
training_epochs = 3
load_previous_weights = True
observe_and_train = True
save_weights = True
num_games_to_play = 1000


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('LunarLander-v2')
env.reset()

#initialize training matrix with random states and actions
dataX = np.random.random(( 5,num_env_variables+num_env_actions ))
#Only one output for the total score
dataY = np.random.random((5,1))



#nitialize the Neural Network with random weights

model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(512, activation='relu', input_dim=dataX.shape[1]))
model.add(Dense(256, activation='relu' ))
model.add(Dense(256, activation='relu'))
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

#Initialize training data array
total_steps = 0
dataX = np.zeros(shape=(1,num_env_variables+num_env_actions))
dataY = np.zeros(shape=(1,1))

#Initialize Memory Array data array
memoryX = np.zeros(shape=(1,num_env_variables+num_env_actions))
memoryY = np.zeros(shape=(1,1))


print("dataX shape", dataX.shape)
print("dataY shape", dataY.shape)


#This function predicts the reward that will result from taking an "action" at a state "qstate"
def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,actions_1_hot[action]), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = model.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward



if observe_and_train:

    #Play the game a determine number of times
    for game in range(num_games_to_play):
        gameX = np.zeros(shape=(1,num_env_variables+num_env_actions))
        gameY = np.zeros(shape=(1,1))
        #Get the initial Q state
        qs = env.reset()
        for step in range (40000):

            #Learn from observation and not playing
            if game < num_initial_observation:
                #take a radmon action
                a = env.action_space.sample()
            else:
                #Now playing and also learning from experience during play

                #Calculate probability to take deterministic action vs random action (epsilon)
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob:
                    #take a random action
                    a=env.action_space.sample()
                    #print("taking random action",a, "at total_steps" , total_steps)
                    #print("prob ", prob, "explore_prob", explore_prob)

                else:
                    ##chose an action by estimating the function-estimator remembered consequences of all possible actions
                    ## Bellman states that the best policy (i.e. action) is the one that maximizez expected rewards for future states
                    ## to caculate rewards we compute the reward a this state t + the discounted (b_discount) reward at all possible state t+1
                    ## all states t+1 are estimated by our function estimator (our Neural Network)


                    utility_possible_actions = np.zeros(shape=(num_env_actions))

                    utility_possible_actions[0] = predictTotalRewards(qs,0)
                    utility_possible_actions[1] = predictTotalRewards(qs,1)
                    utility_possible_actions[2] = predictTotalRewards(qs,2)
                    utility_possible_actions[3] = predictTotalRewards(qs,3)


                    #chose argmax action of estimated anticipated rewards
                    #print("utility_possible_actions ",utility_possible_actions)
                    #print("argmax of utitity", np.argmax(utility_possible_actions))
                    a = np.argmax(utility_possible_actions)



            env.render()
            qs_a = np.concatenate((qs,actions_1_hot[a]), axis=0)

            #print("action",a," qs_a",qs_a)
            #Perform the optimal action and get the target state and reward
            s,r,done,info = env.step(a)


            #record information for training and memory
            if step ==0:
                gameX[0] = qs_a
                gameY[0] = np.array([r])
                memoryX[0] = qs_a
                memoryY[0] = np.array([r])

            gameX = np.vstack((gameX,qs_a))
            gameY = np.vstack((gameY,np.array([r])))


            if done :
                #GAME ENDED
                #Calculate Q values from end to start of game (From last step to first)
                for i in range(0,gameY.shape[0]):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                        gameY[(gameY.shape[0]-1)-i][0] = gameY[(gameY.shape[0]-1)-i][0]
                    else:
                        #print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                        gameY[(gameY.shape[0]-1)-i][0] = gameY[(gameY.shape[0]-1)-i][0]+b_discount*gameY[(gameY.shape[0]-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])
                    if i==gameY.shape[0]-1 and game%5==0:
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

            #Update the states
            qs=s

            #Retrain every X game after num_initial_observation
            if done and game >= num_initial_observation:
                if game%10 == 0:
                    print("Training  game# ", game,"momory size", memoryX.shape[0])
                    model.fit(memoryX,memoryY, batch_size=256,nb_epoch=training_epochs,verbose=0)

            if done:
                if r >= 0 and r <99:
                    print("Game ",game," ended with positive reward ")
                if r > 50:
                    print("Game ", game," WON *** " )
                #Game ended - Break
                break





if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
