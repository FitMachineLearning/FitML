# FitML
```python
*model.fit(Machine_Learning, epochs=Inf)* 
```
[images coming soon]

### What is Fit ML
Fit Machine Learning (FitML) is blog that houses a collection of python Machine Learning articles and examples, often focusing on Reinforcement Learning. Here, you will find code related to Q Learning, Actor-Critic, MDP, Bellman, OpenAI solutions and custom implemented approaches to solving some of the toughest and most interesting problems to date (Yes, I am "baised").

### Who is Michel Aka
*Michel is an AI researcher and a graduate from University of Montreal who currently works in the Healthcare industry.*

### RL Approaches

#### Optimal Policy Tree Search

This is a RL technique which is characterized by computing the estimated value of expected sum of reward for n time steps ahead. This technique has the advantage of yeilding a better estimation of taking a specific policy, however it is computationally expensive and memorry inneficient. If one had a super computer and very large amount of memory, this technique would do extremely well for discrete action space problem/environments. I believe Alfa-Go uses a varient of this technique.

See examples and find out more about Optimal Policy Tree Search here.

#### Selective Memory

As far as I know, I haven't seen anyone in the litterature implement this technique before.

The intuition behind Policy Gradient is that it optimizes the parameters of the network in the direction of higher expected sum of rewards. What if we could do the same in a computationally more effective way that also turns out to be more intuitive: enter what I am calling Selective Memory.

1)Our objective here is to ensure that the Policy function approximator tends to higher rewards. 

2) We know that Neural Networks will converge towards assigned labeled of our data set and will also generalize (function approximation). 

3)What if there was a way to select our training (reinforcement) data set so that it ensures that we converge towards our objective; Higher expected rewards.

What if we selectively remember actions based on the how high a reward was. In other words, the probability *P* of recording an action state into memory is dependent on the actual sum of reward yeilded by this action trajectory. (Notice that we are not using the expected sum of reward here).

What does this look like in code

First we calculate sum of rewards at the end of each rollout using bellman.

Then we careful select what we want to remember i.e. store in memory
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

#### Q-Learning

Q-Learning is a well knon Reinforcement Learning approach, popularized by Google Deep Mind, when they used it to master multiple early console era games. Q-Learning focuses on estimating the expected sum of rewards using the Bellman equation in order to determine which action to take. Q-Learning works especially well in discrete action space and on problems where the *f(S)->Q* is differentiable, this is not always the case.

Find out more about Q-Learning here.


#### Actor Critique Approaches

Actor Critique is an RL technique which combines Policy Gradient appraoch with a Critique (Q value estimator)

Find out more about Actor-Critique here.

### Recommended Progression for the Newcomer

[coming soon]

###

### Code Samples

<table style="width:100%">
  <tr>
    <th>Approach</th>
    <th>Examples</th> 
  </tr>
  <tr>
    <td>CNN</td>
    <td>[coming soon]</td> 
  </tr>
  <tr>
    <td>Optimal Policy Tree Search</td>
    <td><a href ="https://github.com/FitMachineLearning/FitML/blob/master/OptimalPolicyTreeSearch/Cartpole_OPTS.py">Cartpole_OPTS.py</a> </td> 
  <tr>
    <td>Selective Memory</td>
    <td><a href ="https://github.com/FitMachineLearning/FitML/blob/master/SelectiveMemory/CartPole_SelectiveMemory.py">CartPole_SelectiveMemory.py</a>
      <BR>
          <a href ="https://github.com/FitMachineLearning/FitML/blob/master/SelectiveMemory/LunarLander_Selective_Memory.py">LunarLander_Selective_Memory.py</a>
          <BR>
          <a href ="https://github.com/FitMachineLearning/FitML/blob/master/SelectiveMemory/BipedalWalker_Selective_Memory.py">BipedalWalker_Selective_Memory.py</a>
          </td>       
  </tr>
  <tr>
    <td>Q-Learning / Deep-QN</td>
    <td>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/DeepQN/CartPole_QLearning.py">CartPole_DeepQN.py</a> 
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/DeepQN/LunarLander_QL.py">LunarLander_DQN.py</a>   
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/DeepQN/Atari_Pong_DeepQN.py">Atari_Pong_DeepQN.py</a> 
    </td>      
  <tr>        
</table>
