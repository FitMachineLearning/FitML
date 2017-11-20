# FitML
*model.fit(Machine_Learning, epochs=Inf)* 

A collection of python Machine Learning articles and examples. Here, you will find code related to Q Learning, Actor-Critic, MDP, Bellman, OpenAI solutions and others...

*Michel is an AI researcher and a graduate from University of Montreal*

### Selective Memory

This new approach, that I am naming Selective Memory, optmizes what to remember based on how good an action was. This approach has yeilded significantly improved results compared to Q-Learning and Actor Critic.

It is able to solve CartPole by only using 10 samples and balancing the beam for all future itteration after the 15th try.
You can find the code here.

https://github.com/FitMachineLearning/FitML/blob/master/CartPole_SelectiveMemory.py

There is also an implementation of Lunar Lander with our Selective Memory Approach

https://github.com/FitMachineLearning/FitML/blob/master/LunarLander_Selective_Memory.py

First we calculate sum of rewards at the end of each rollout using bellman.

The we careful select what we want to remember i.e. store in memory
```python
def addToMemory(reward,rangeL,rangeH):
    prob = reward - rangeL
    prob = prob / (rangeH - rangeL)
    if np.random.rand(1)<=prob :
        #print("Adding reward",reward," based on prob ", prob)
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

When we get a new state we then act based on optimal policy which has been trained on memory primed with only the best results yeilding actions.
```python
    #Get Remembered optiomal policy
    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)
    a = remembered_optimal_policy
```



### Great explanation of Policy Gradient and Deterministic Policy Gradient by David Silver
This one is a must watch

http://techtalks.tv/talks/deterministic-policy-gradient-algorithms/61098/

### OpenAI Lunar Lander with Modified Q Learning

<a href="https://www.youtube.com/watch?v=p0rGjAgykOU"><img src="/img/LunarLandQLearning.png" width="350"/><a>

Implementation of a Q Learning Algorithm on the OpenAI LunarLander. 
After 150 itterations the agent can more or less fly safely.
After 400 itterations the agent is able to land safely most of the time.
After 600 itterations the agent is able to land safely on the pad the majority of the time.

Demo of the agent can be seen here

https://www.youtube.com/watch?v=p0rGjAgykOU

You can find the code here

https://github.com/FitMachineLearning/FitML/blob/master/LunarLander_QL.py

In our example we use the following
* Markov Decision Process
* Bellman equation
* Reinforcement Learning with epsilon discovery probability reduced over time
* Q / utility estimation with NN
* RL Memory

You can play with the parameters to experiment.

### Lunar Lander using Actor Critic

This solution of the OpenAi LunarLander uses 2 networks. One action predictor (commonly referred to as actor) and one future reward predictor (commonly referred to as critic). It is computationaly more effective than the recursive tree search, but takes longer to solve the problem.

https://github.com/FitMachineLearning/FitML/blob/master/LunarLander_ActorCritic.py

You can see the agent in action here:

https://www.youtube.com/watch?v=z9R5hDT6vUQ


Note that this method takes longer to solve the problem. Higher rates of successful landing start to appear after 1200 tries. There are definitely optimizations that can be brought by playing with
1) Learning rate
2) Monte-Carlo randomness i.e. number of random samples polled for Q value comparisons
3) Network width / Number of Neurons
4) ... Many others

On thing to note, this agent exhibits more complex behaviors (i.e. more tricks) than its Q Learning counterpart. 

Optimal policy is chosen by taking highest returned function-estimator Q value between remembered action and random sample.

```python
if predictTotalRewards(qs,remembered_optimal_policy) > predictTotalRewards(qs,randaction):
    a = remembered_optimal_policy
else
```

#### Understanding Actor Critic

The best way to think about Actor Critic is to have an intuitive understanding of it first.

The Actor always tries to predict the best action based on experience.
The Critic is responsible for determining which policies, taken from a sample, yield the best long-term reward. 
Here is a write up of a typical scenario in Actor Critic epoch

* Env: "Hey we have a new state, it's called s' ."
* Actor: "Oh, Oh, I know the best action to take. I remember, I've seen this before... We should do Action aa' "
* Critic: "Not so fast buddy, let me check first if aa' is better than these other actions".
* Actor: "But, but, these samples are random, there is no way that ..."
* Critic: "Well according to my calculations Sample A3 with action a3' yields the highest long term reward."
* Actor: "Dude... "
* Critic: "Trust me. I learn faster than you" (It is optimal for critic to have a higher learning rate)
* Actor: "Ok.. You're the boss" (Actor adjusts it's optimal action for this state to Sample Action a3' )
* Env: "After taking action a3', you got reward R'' "
* Critic: "Oh, my.... I was wrong. That's Ok, I'll do better next time (Critic adjust it's weights to converge towards R'' for this state action (s'+a3')

In essence, the Actor always improves towards what the Critic thinks is best at that moment in time. The Critic keep getting better understanding of its environment every iteration. The Actor provides the adventage of a state dependant memory. If the Actor wasn't there we would always have to select from randon; And this is not guarantied to display expected behavior.

There are multiple strategies to chose the sample actions to be evaluated each epochs by the Critic. But one thing remains. Eventually, the Actor will learn to exceed the long-term reward at all epochs of random selected sample actions (or actions selected otherwise). The Critic will understand how reward works within the environment and the Actior will remember or generalize to figure out what action to take in any given situation.


I highly recommend reading the Actor-Critic Algorithm paper from Vijay Konda and John Tsitsiklis.
https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf




### CartPole with function estimator recursive Tree Search Bellman reward computation 

[![Cartpole Demo](https://img.youtube.com/vi/TguWjWvRp8c/0.jpg)](https://www.youtube.com/watch?v=TguWjWvRp8c)


#### Results
After observing the environment for 30 episode our agent is able to balance the pole for all subsequent games indefinitely. We get the best results by anticipating 4-8 steps in advance. The more the agent can anticipate (larger number of predicted steps) the more stable the pole is balanced. You can see how it behaves with different depth settings in the video link below. 

https://www.youtube.com/watch?v=TguWjWvRp8c



#### Approach

We use a function approximator and teach it to predict/estimate the next state and reward pair base on current state and action. Once our function approximator learns the relationship betwenn (s',a') and (s'',r'') we then use it to recusively calculate estimated longterm reward using bellman for x future steps.


Find the code here: https://github.com/FitMachineLearning/FitML/blob/master/Cartpole_MDP.py


### CarPole with Actor Critic

This solution of the CartPole uses 2 networks. One action predictor (commonly referred to as actor) and one future reward predictor (commonly referred to as critic). It is computationaly more effective than the recursive tree search, but takes longer to solve the problem.

https://github.com/FitMachineLearning/FitML/blob/master/CartPole_ActorCritic.py

### CartPole with Modified Q Learning

This solution of the CartPole uses a function approximator to estimate the Q/Utilities of all possible future states.
(qs',a') => (R'').

```python
utility_possible_actions[0] = predictTotalRewards(qs,0)
utility_possible_actions[1] = predictTotalRewards(qs,1)
```
We then select the policy/action with the highest estimated Q value.

### Experimental A2C algorithm for Bipedal Walker

Find the code here. I would be interested to see what results you guys get if you happen to play with the parameters.
https://github.com/FitMachineLearning/FitML/blob/master/BepedalWalker_A2C.py

