# FitML
```python
model.fit(Machine_Learning, epochs=Inf)
```

<table style="width:100% border: none" >
  <tr>
    <th><img src="/img/cWalker.jpg" width="250"/></th>       
  </tr>
</Table>
 
https://youtu.be/hKrFFeZqq3E

#### How does Selective Memory work?

The intuition behind Policy Gradient is that it optimizes the parameters of the network in the direction of higher expected sum of rewards. What if we could do the same in a computationally more effective way that also turns out to be more intuitive: enter what I am calling Selective Memory.

1) Our objective here is to ensure that the Policy function converges towards higher rewards. 

2) We know that Neural Networks will converge towards assigned labeled of our data set and will also generalize (function approximation). 

3) What if there was a way to select our training (reinforcement) data set so that it ensures that we converge towards our objective; Higher expected rewards.

Here we propose the approach of selectively remembering actions based on the how high a reward was. In other words, the probability *P* of recording an action state into memory (or a rollout) is dependent on the actual sum of reward yeilded by this action trajectory. (Notice that we are not using the expected sum of reward here but the actual computed value at the end of the rollout).

What does this look like in code

First we creat our function approximators Neural Networks
```python
#nitialize the Reward predictor model
model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(1024, activation='relu', input_dim=dataX.shape[1]))
model.add(Dense(256, activation='tanh'))
model.add(Dense(dataY.shape[1]))
opt = optimizers.adam(lr=learning_rate)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(1024, activation='relu', input_dim=apdataX.shape[1]))
action_predictor_model.add(Dense(512, activation='relu'))
action_predictor_model.add(Dense(apdataY.shape[1],activation='tanh'))
```

Then we calculate sum of rewards at the end of each rollout using Bellman.

Then we careful select what we want to remember i.e. store in memory.

There is a number of approaches we have used to discriminate on the nature of the State-Actions or State-Action-Rewards that we will be keeping in memory to train our Actor. One discriminates for each indivudual action state, the other discriminates an entire rollout batch. Reguardless the principle is the same. We determine how good an action is compared to the average remembered good actions.

```python
def addToMemory(reward,averageReward):

    prob = 0.1
    if( reward > averageReward):
        prob = prob + 0.9 * math.tanh(reward - averageReward)
    else:
        prob = prob + 0.1 * math.tanh(reward - averageReward)

    if np.random.rand(1)<=prob :
        print("Adding reward",reward," based on prob ", prob)
        return True
    else:
        return False
```

```python
    for i in range(0,gameR.shape[0]):
        if addToMemory(gameR[i][0],-1,50):
            tempGameSA = np.vstack((tempGameSA, gameSA[i]))
            tempGameA = np.vstack((tempGameA,gameA[i]))
            tempGameR = np.vstack((tempGameR,gameR[i]))
            tempGameS = np.vstack((tempGameS,gameS[i]))
```

Here gameSA, gameA, gameR and gameS represent the various State-Action pairs, Actions, actual discounted sum of rewards and States respectively.

When we get a new state we then act based on optimal policy which has been trained on memory primed with only the best results yeilding actions.
```python
    #Get Remembered optiomal policy
    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)
    a = remembered_optimal_policy
```

### What type of results do we get?
Our agent is able to crawl, stand up, walk, run, jump after 500 episodes in the famous openAI BipedalWalker test. After 3000 iterations, our agent is able to advance fast and be very stable on its feet.
You can watch it in action here: https://youtu.be/hKrFFeZqq3E.


### What is Fit ML
Fit Machine Learning (FitML) is blog that houses a collection of python Machine Learning articles and examples, often focusing on Reinforcement Learning. Here, you will find code related to Q Learning, Actor-Critic, MDP, Bellman, OpenAI solutions and custom implemented approaches to solving some of the toughest and most interesting problems to date (Yes, I am "baised").

### Who is Michel Aka
*Michel is an AI researcher and a graduate from University of Montreal who currently works in the Healthcare industry.*
