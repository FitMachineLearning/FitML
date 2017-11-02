'''
Cartpole solution by the Author of the Fit Machine Learning Blog

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
num_env_actions = 1
num_training_exmaples = 100
timesteps = 1
num_initial_observation = 4000
training_epochs = 300
num_anticipation_steps = 6
load_previous_weights = True
observe_and_train = False
save_weights = False


#Create testing enviroment
env = gym.make('CartPole-v0')
env.reset()

#initialize training matrix with random states and actions
dataX = np.random.random(( num_training_exmaples,num_env_variables+num_env_actions ))
#initize training matrix corresponding expected states and expected rewards (random)
dataY = np.random.random((num_training_exmaples,num_env_variables+1))



#nitialize the LSTM with random weights

model = Sequential()
model.add(LSTM(16,return_sequences=True, stateful=True , batch_size=1,  input_shape=(timesteps, dataX.shape[1])))
model.add(LSTM(16, return_sequences=True))
model.add(Dense(16, activation='relu'))
model.add(Dense(dataY.shape[1]))

opt = optimizers.adam(lr=0.01)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

#load previous model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/CP-weights.h5"
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        model.load_weights("CP-weights.h5")
    else:
        print("File CP-weights.h5 does not exis. Retraining... ")

#Record first 500 in a sequence and add them to the training sequence
total_steps = 0
dataX = np.zeros(shape=(1,5))
dataY = np.zeros(shape=(1,5))

print("dataX shape", dataX.shape)
print("dataY shape", dataY.shape)

if observe_and_train:
    #observe for 100 games
    for game in range(100):

        if total_steps >= num_initial_observation:
            break
        #Get the Q state
        qs = env.reset()
        for step in range (200):
            a=0
            if np.random.rand(1) < 0.5:
                a=0
            else:
                a=1
            env.render()
            qs_a = np.concatenate((qs,np.array([a])), axis=0)

            #get the target state and reward
            s,r,done,info = env.step(a)

            #set reward in case of failure
            if done:
                r = -1

            #concatenate target state and reward
            s_r = np.concatenate((s,np.array([r])), axis=0)

            if done:
                #print negative reward array
                print("Negative reward s_r: ", s_r)

            #print("reward = ", r)
            #print("target state", s)
            #print("concatenate(s,r)", s_r)


            #record only the first x number of states
            if total_steps ==0:
                dataX[0] = qs_a
                dataY[0] = s_r

            if total_steps < (num_initial_observation-1):
                dataX = np.vstack((dataX,qs_a))
                dataY = np.vstack((dataY,s_r))

            #Update the states
            qs=s


            total_steps += 1
            if done :
                break

    print("Observation complete. - Begin LSTM training")

    print("dataX shape", dataX.shape)
    print(dataX[0:5])
    print("dataY shape", dataY.shape)
    print(dataY[0:5])

    feedX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1] ))
    feedY = np.reshape(dataY, (dataY.shape[0], 1, dataY.shape[1] ))


    #The more epochs you train the model, the better is becomes at predicting future states
    #This in turn will improve the results of the Bellman equation and thus will lead us to
    # better decisions in our MDP process
    model.fit(feedX,feedY, batch_size=1,epochs=training_epochs,verbose=2)

    print("total_steps ", total_steps)
    print("dataX ", dataX[0:10])
    print("dataY ", dataY[0:10])
    #print("dataY ", dataY)

dataX = np.random.random((1,5))

res = model.predict(dataX[0].reshape(1,1,dataX.shape[1]))
nstate = res[0][0][:-1]

print("predicted output ", res)
print("expected reward ", res[0][0][4])
print("expected state ", nstate)

def estimateReward(qstate,action, depth):
    if depth <= 0:
        return 0
    #calculate/estimate reward at this state and get the next state
    qs_a = np.concatenate((qstate,np.array([action])), axis=0)
    predX = np.zeros(shape=(1,5))
    predX[0] = qs_a
    pred = model.predict(predX[0].reshape(1,1,predX.shape[1]))
    reward = pred[0][0][4]
    expected_state = pred[0][0][:-1]

    '''
    print("depth -- ", depth)
    print("qstate", qstate)
    print("action", action)
    print("pred", pred)
    print("expected_state", expected_state)
    print("reward", reward)
    '''
    # Bellman -- reward at this state = reward + Sum of discounted expected rewards for all actions (recursively)
    #recursively calculate the reward at future states for all possible actions
    discounted_future_rewards = 0.95*estimateReward(expected_state,0,depth-1)+ 0.95*estimateReward(expected_state,1,depth-1)

    #print("discounted_future_rewards", discounted_future_rewards)
    #add current state and discounted future state reward
    return reward + discounted_future_rewards


print("** Estimating reward for dataX[0] with action 1 usint Bellman", estimateReward(dataX[0][:-1],1,2))
print("** Estimating reward for dataX[0] with action 0 usint Bellman", estimateReward(dataX[0][:-1],0,2))



#####
#####
#Play the game for X rounds using the Bellman with LSTM anticipation model


for game in range(3):
    total_steps =0
    #Get the Q state
    qs = env.reset()
    for step in range (300):
        ##chose an action by estimating consequences of actions for the next num_anticipation_steps steps ahead
        #works best with looking 6 steps ahead
        #Also works best if you train the model more itterations
        estimated_anticipated_reward_a = estimateReward(qs,1,num_anticipation_steps)
        estimated_anticipated_reward_b = estimateReward(qs,0,num_anticipation_steps)
        #print(" estimated rewards a and b", estimated_anticipated_reward_a, estimated_anticipated_reward_b)

        #chose argmax action of estimated anticipated rewards
        if estimated_anticipated_reward_a > estimated_anticipated_reward_b:
            a = 1
        else:
            a = 0

        env.render()


        #get the target state and reward
        s,r,done,info = env.step(a)



        qs=s
        #set reward in case of failure
        if done:
            r = -1
            if total_steps >= 198:
                print("*** Game Won after ", total_steps, " steps")
            else:
                print("** failed after ", total_steps, " steps")


        total_steps += 1
        if done :
            break

if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights("CP-weights.h5")
