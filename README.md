# FitML
```python
model.fit(Machine_Learning, epochs=Inf)
```
<table style="width:100% border: none" >
  <tr>
    <th><img src="/img/cCartPole.jpg" width="250"/></th>      
    <th><img src="/img/LunarLandQLearning.png" width="250"/></th>
    <th><img src="/img/cWalker.jpg" width="250"/></th> 
    <th><img src="/img/cPong.jpg" width="250"/></th>      
  </tr>
</Table>

### What is Fit ML
Fit Machine Learning (FitML) is blog that houses a collection of python Machine Learning articles and examples, often focusing on Reinforcement Learning. Here, you will find code related to Q Learning, Actor-Critic, MDP, Bellman, OpenAI solutions and custom implemented approaches to solving some of the toughest and most interesting problems to date (Yes, I am "baised").

### Who is Michel Aka
*Michel is an AI researcher and a graduate from University of Montreal who currently works in the Healthcare industry.*

### RL Approaches

#### Optimal Policy Tree Search

This is a RL technique which is characterized by computing the estimated value of expected sum of reward for n time steps ahead. This technique has the advantage of yeilding a better estimation of taking a specific policy, however it is computationally expensive and memorry inneficient. If one had a super computer and very large amount of memory, this technique would do extremely well for discrete action space problem/environments. I believe Alfa-Go uses a varient of this technique.

See examples and find out more about Optimal Policy Tree Search <a href="https://github.com/FitMachineLearning/FitML/tree/master/OptimalPolicyTreeSearch"> here </a>.

#### Selective Memory

As far as I know, I haven't seen anyone in the litterature implement this technique before.

The intuition behind Policy Gradient is that it optimizes the parameters of the network in the direction of higher expected sum of rewards. What if we could do the same in a computationally more effective way that also turns out to be more intuitive: enter what I am calling Selective Memory. 

We chose what to commit to memory based on actual sum of rewards

Find out more <a href="https://github.com/FitMachineLearning/FitML/tree/master/SelectiveMemory"> here </a>.


#### Q-Learning

Q-Learning is a well knon Reinforcement Learning approach, popularized by Google Deep Mind, when they used it to master multiple early console era games. Q-Learning focuses on estimating the expected sum of rewards using the Bellman equation in order to determine which action to take. Q-Learning works especially well in discrete action space and on problems where the *f(S)->Q* is differentiable, this is not always the case.

Find out more about Q-Learning <a href="https://github.com/FitMachineLearning/FitML/tree/master/DeepQN"> here </a>.


#### Actor Critique Approaches

Actor Critique is an RL technique which combines Policy Gradient appraoch with a Critique (Q value estimator)

Find out more about Actor-Critique <a href="https://github.com/FitMachineLearning/FitML/tree/master/ActorCritic"> here </a>.

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
    <td>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/SelectiveMemory/CartPole_SelectiveMemory.py">CartPole_SelectiveMemory.py</a>
        <BR>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/SelectiveMemory/LunarLander_Selective_Memory.py">LunarLander_Selective_Memory.py</a>
         <BR>
         <a href ="https://github.com/FitMachineLearning/FitML/blob/master/SelectiveMemory/BipedalWalker_Selective_Memory.py">BipedalWalker_Selective_Memory.py</a>
     </td>       
  </tr>
  <tr>
    <td>Q-Learning / Deep-QN</td>
    <td>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/DeepQN/CartPole_QLearning.py">CartPole_DeepQN.py</a> <BR>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/DeepQN/LunarLander_QL.py">LunarLander_DQN.py</a>   <BR>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/DeepQN/Atari_Pong_DeepQN.py">Atari_Pong_DeepQN.py</a> 
    </td>      
  <tr>
    
  <tr>
    <td>Actor Critic</td>
    <td>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/ActorCritic/CartPole_ActorCritic.py">CartPole_AC.py</a> <BR>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/ActorCritic/LunarLander_ActorCritic.py">LunarLander_AC.py</a>   <BR>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/ActorCritic/BepedalWalker_A2C.py">BipedalWalker_AC.py</a> <BR>
        <a href ="https://github.com/FitMachineLearning/FitML/blob/master/ActorCritic/Acrobot_ActorCritic.py">Acrobot_AC.py</a>      
    </td>      
  <tr>   

</table>
