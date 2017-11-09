# FitML
*model.fit(Machine_Learning, epochs=Inf)* 

A collection of python Machine Learning articles and examples. Here, you will find code related to MDP, Bellman, OpenAI solutions and others...

*Michel is an AI researcher and a graduate from University of Montreal*

## OpenAI Lunar Lander with Modified Q Learning

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

## Lunar Lander using Actor Critic

This solution of the OpenAi LunarLander uses 2 networks. One action predictor (commonly referred to as actor) and one future reward predictor (commonly referred to as critic). It is computationaly more effective than the recursive tree search, but takes longer to solve the problem.

https://github.com/FitMachineLearning/FitML/blob/master/LunarLander_ActorCritic.py

Note that this method takes longer to solve the problem. Higher rates of successful landing start to appear after 1200 tries. There are definately optimizaations that can be brought by playing with
1) Learning rate
2) Monte-Carlo randomness i.e. number of random samples polled for Q value comparisons
3) Network width / Number of Nerones
4) ... Many others

On thing to note, this agent exhibits more complex behaviors (i.e. more tricks) than its Q Learning counterpart. 

Optimal policy is chosen between highest function-estimator Q value between remembered action and random sample.

```python
if predictTotalRewards(qs,remembered_optimal_policy) > predictTotalRewards(qs,randaction):
    a = remembered_optimal_policy
else
```

## CartPole with recursive Tree Search Bellman reward computation 

[![Cartpole Demo](https://img.youtube.com/vi/TguWjWvRp8c/0.jpg)](https://www.youtube.com/watch?v=TguWjWvRp8c)


### Results
After observing the environment for 30 episode our agent is able to balance the pole for all subsequent games indefinitely. We get the best results by anticipating 4-8 steps in advance. The more the agent can anticipate (larger number of predicted steps) the more stable the pole is balanced. You can see how it behaves with different depth settings in the video link below. 

https://www.youtube.com/watch?v=TguWjWvRp8c



### Approach

We use a function approximator and teach it to predict/estimate the next state and reward pair base on current state and action. Once our function approximator learns the relationship betwenn (s',a') and (s'',r'') we then use it to recusively calculate estimated longterm reward using bellman for x future steps.


Find the code here: https://github.com/FitMachineLearning/FitML/blob/master/Cartpole_MDP.py


## CarPole with Actor Critic

This solution of the CartPole uses 2 networks. One action predictor (commonly referred to as actor) and one future reward predictor (commonly referred to as critic). It is computationaly more effective than the recursive tree search, but takes longer to solve the problem.

https://github.com/FitMachineLearning/FitML/blob/master/CartPole_ActorCritic.py

## CartPole with Modified Q Learning

This solution of the CartPole uses a function approximator to estimate the Q/Utilities of all possible future states.
(qs',a') => (R'').

```python
utility_possible_actions[0] = predictTotalRewards(qs,0)
utility_possible_actions[1] = predictTotalRewards(qs,1)
```
We then select the policy/action with the highest estimated Q value.

