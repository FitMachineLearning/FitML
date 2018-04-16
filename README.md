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

### How to use for Reinforcement Learning Algorithm
- (Optional) Clone the repo 
- Select the algorithm that you need (Folders are named by the RL algorithm ). Policy Gradient/ Parameter Noising/ Actor Critic / Selective memory
- Get an instance of the algorithm with the environment you need. If the one you are looking for isn't there, get any environment.py file from the algorithm folder of choice and follow the steps below.
- Install the dependencies
- - Usually "pip install <library name>". Example "pip install pygal"
- Replace the name of the environment in line 81 of the code.
 ```Python
  env = gym.make('BipedalWalker-v2')
  # replace with
  env = gym.make('<your-environement-name-here>')
 ```
   or set the ```ENVIRONMENT_NAME =``` to your environment name. Example ```ENVIRONMENT_NAME = "BipedalWalker-v2"```.
  
- set the environment's observation and action space and viriables. If you don't know them, run the script once and they will be printed in the first lines of your output.
 ```Python
  num_env_variables = <number of observation variables here>
  num_env_actions = <number of action variables here>
 ```
- (Optional) you can check the results of your agent as it progresses with the .svg file in the same directory as your script. Any modern browser can view them. 

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


