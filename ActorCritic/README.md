# Solving Bipedal Walker with Actor Critic with Python and Keras
```python
model.fit(Machine_Learning, epochs=Inf)
```
<table style="width:100% border: none" >
  <tr>
    <th><img src="/img/Walker.png" height="250" align="center"/></th> 
  <th><img src="/img/DeepQN.png" height="250" align="center"/></th> 
  </tr>
</Table>

### What is Fit ML
Fit Machine Learning (FitML) is blog that houses a collection of python Machine Learning articles and examples, often focusing on Reinforcement Learning. Here, you will find code related to Q Learning, Actor-Critic, MDP, Bellman, OpenAI solutions and custom implemented approaches to solving some of the toughest and most interesting problems to date (Yes, I am "baised").

### What is Bipedal Walker anyway?
Bipedal Walker is an <a href="https://openai.com/systems/">OpenAI Gym</a> environment where an agent learns to control a bipedal walker in order to reach the end of an obstacle course. What makes this challenging is that 
1) The agent only receives limbs coordinates along with Lidar information
2) Actions are vectors of 4 real numbers
So our agent has to learn to balance,walk,run,jump on its own without any human intervention.

### Why Q-Learning alone doesn't work
For those acquinted with QLearning, it becomes clear very quickly that we cannot apply a greedy policy here. Simply relying on a Q-value function approximator and polling on non-discrete action space, let along a vector of of continuous action space is simply impossible. In order to overcome this challenge we will the Actor Critic Method where 1 Nerual Network is in charge of approximating how good an action is, and the other learns what to do in any given situation.

Let's see how this is implemented using keras.

### Creating The Actor and the Critic
Since we don't know how good an action is going to be until such time that we have take it, a common technique in Reinforcement Learning is to predict/approximate this using a function approximator a.k.a. a Neural Network. We will call this first network QModel. It will take as input a combination of state-action and estimate how good this is.

```Python
#nitialize the Reward predictor model
Qmodel = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(4096, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(dataY.shape[1])) #dataY.shape[1] is 1 corresponding to the single Real approximated value

```

We now need to ensure that we have a way to act optimally at every state. This is where the Actor comes in. This is another function approximator that takes a state as input and outputs an action.

### Helper functions
We then declare a set of helper functions that are going to be use to optimze our actions at every state.

```Python
def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = Qmodel.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward


def GetRememberedOptimalPolicy(qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy
```

# Exploration
As we initally have no concept of optimal policy, we need to ensure that some actions are taken stochastically. This will prevent our model from stagnating in its improvement.

```python
    prob = np.random.rand(1)
    explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

    #Chose between prediction and chance
    if prob < explore_prob:
        #take a random action
        a = env.action_space.sample()
 ```


#### Very good course on Actor Critic
http://mi.eng.cam.ac.uk/~mg436/LectureSlides/MLSALT7/L6.pdf

### Who is Michel Aka
*Michel is an AI researcher and a graduate from University of Montreal who currently works in the Healthcare industry.*
