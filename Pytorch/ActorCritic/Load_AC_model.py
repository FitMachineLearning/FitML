## DQN Tutorial
## https://www.youtube.com/watch?v=WHRQUZrxxGw
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Any
from random import random

@dataclass
class sars:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    qval: float

class DQNAgent:
    def __init__(self,actor_model,critic_model):
        self.actor_model = actor_model
        self.critic_model = critic_model

    def get_actions(self, observations):
        # import ipdb; ipdb.set_trace()
        guessed_actions = self.actor_model(torch.Tensor(observations).to(self.actor_model.device))
        return guessed_actions

    def get_predicted_Q_values(self,observation_and_action):
        guessed_Qs = self.crtic_model(torch.Tensor(observation_and_actioner))
        return guessed_Qs(-1)[1]

    def update_target_model(self):
        self.targetModel.load_state_dict(self.model.state_dict())


class ActorModel(nn.Module):
    def __init__(self, obs_shape, action_shape,lr):
        super(ActorModel,self).__init__()
        assert len(obs_shape) ==1, "This network only works on flat observations"
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # import ipdb; ipdb.set_trace()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0],256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,action_shape[0])
        )
        self.opt = optim.Adam(self.net.parameters(),lr=lr)
        if torch.cuda.is_available():
            print("Using CUDA")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, x):
        return self.net(x)


class CriticModel(nn.Module):
    def __init__(self, obs_shape, action_shape,lr):
        super(CriticModel,self).__init__()
        assert len(obs_shape) ==1, "This network only works on flat observations"
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0]+action_shape[0],512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,1) # one out put because we are predicting Q values
        )
        self.opt = optim.Adam(self.net.parameters(),lr=lr)
        if torch.cuda.is_available():
            print("Using CUDA")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, buffer_size = 1000):
        self.buffer_size = buffer_size
        # self.buffer = [None]*buffer_size
        self.buffer = []
        self.index = 0

    def insert(self, sars):
        # self.buffer.append(sars)
        # print("inserting index ", self.index, "@",self.index%self.buffer_size)
        if(self.index == 10):
            print("first 10 ",self.buffer[0:10])
            # import ipdb; ipdb.set_trace()

        # if(self.index > self.buffer_size and self.index%self.buffer_size==0):
        #     print("first 10 ",self.buffer[0:10])
        #     print("last 10 ",self.buffer[-10:])
        #     print("")
        #     import ipdb; ipdb.set_trace()
        # self.buffer[self.index%self.buffer_size] = sars
        self.index+=1
        self.buffer.append(sars)
        if(len(self.buffer)>self.buffer_size):
            self.buffer = self.buffer[1:]
            # print("Clipping Buffer at size", len(self.buffer))

    def sample(self, num_samples,current_episode_steps):
        # assert num_samples < min(len(self.buffer),self.index)
        # if num_samples>self.index:
        # print("sampling n ",min(num_samples,self.index))
        # a = self.buffer[0:((self.index-current_episode_steps)%self.buffer_size)]
        if len(self.buffer) > 0:
            return np.random.choice(self.buffer, min(num_samples,self.index))
        else:
            return []



def get_one_hot(action,n_dim):
    retval = np.zeros(n_dim)
    retval[action] = 1.0
    return retval


def train_actor(actor_model, critic_model, state_transitions, num_actor_training_samples, num_actions):
    #for each observation get the critic to generate the Q value corresponding to each action_space
    #retain action observation pairs corresponding to the highest Q values
    #train the actor to converge towards that set

    #Generate random actions
    random_actions = []
    for i in range(num_actor_training_samples):
        random_actions.append( np.random.rand(num_actions)*2-1 )
    #Get random observations
    random_states = [s.state for s in state_transitions]

    # import ipdb; ipdb.set_trace()

    # for earch state add the best corresponding action to random actions
    for i in range(len(random_states)):
        with torch.no_grad():
            act = actor_model(torch.Tensor(random_states[i]).to(actor_model.device)) .cpu().detach().numpy()
            random_actions.append(act)



    best_state_action = []
    for i_states in range(len(random_states)):
        QAs = []

        # get the Qvalues from the random actions
        for i_actions in range(len(random_actions)):
            with torch.no_grad():
                qval = critic_model( torch.Tensor(   torch.cat( (torch.Tensor(random_states[i_states]),torch.Tensor(random_actions[i_actions])),0 )    ).to(critic_model.device) ).cpu()
                QAs.append( qval )
        # get index for best actions between all random actions and the actor's predicted actions
        #_sars = sars(observation,action,reward,observation_next,done,0.0)
        best_state_action.append(sars(random_states[i_states], random_actions[np.argmax(QAs)],0.0,None,False,np.max(QAs) ))
    # import ipdb;ipdb.set_trace()

    t_random_states = torch.stack( ([torch.Tensor(s.state) for s in best_state_action]) ).to(actor_model.device)
    target_actions = torch.stack( ([torch.Tensor(s.action) for s in best_state_action]) ).to(actor_model.device)
    actor_model.zero_grad()
    predicted_actions = actor_model(t_random_states)

    # loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), actual_Q_values.view(-1) )
    loss = F.smooth_l1_loss(predicted_actions, target_actions ).mean()
    loss.backward()
    actor_model.opt.step()
    return loss

def train_critic(critic_model, state_transitions, num_actions):
    if len(state_transitions) <=0:
        print("empty state transitions")
        return


    cur_states = torch.stack( ([torch.Tensor(torch.cat((torch.Tensor(s.state),torch.Tensor(s.action)),0)) for s in state_transitions]) ).to(critic_model.device)


    rewards = torch.stack( ([torch.Tensor([s.reward]) for s in state_transitions]) ).to(critic_model.device)
    Qs = torch.stack( ([torch.Tensor([s.qval]) for s in state_transitions]) ).to(critic_model.device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(critic_model.device)
    next_states = torch.stack( ([torch.Tensor(s.next_state) for s in state_transitions]) ).to(critic_model.device)
    actions = [s.action for s in state_transitions]
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        actual_Q_values = Qs
        # pred_qvals_next = critic_model(next_states)[0]
    critic_model.opt.zero_grad()
    pred_qvals = critic_model(cur_states)

    # one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions).to(model.device)
    # loss = torch.mean(torch.sqrt((torch.sum(pred_qvals*one_hot_actions,-1) - actual_Q_values.view(-1) )**2)).to(model.device)
    loss = F.smooth_l1_loss(pred_qvals.view(-1), actual_Q_values.view(-1) )
    # loss = F.smooth_l1_loss(torch.sum(pred_qvals,-1), rewards.view(-1)+0.98*mask[:,0]*pred_qvals_next.view(-1) ).mean()
    loss.backward()
    critic_model.opt.step()
    return loss

def update_Qs(replay_buffer,step_counter,episode_len,buffer_size):
    for i in range(episode_len):
        # if(step_counter > buffer_size):
        # import ipdb; ipdb.set_trace()
        index = episode_len-i
        next_index = index+1
        if i==0:
            replay_buffer[index].qval = replay_buffer[index].reward
            if(step_counter%2000==0):
                print("i",i,"q ",replay_buffer[index].qval)
        else:
            replay_buffer[index].qval = replay_buffer[index].reward + 0.98 * replay_buffer[next_index].qval
            if(step_counter%2000==0):
                print("i",i,"q ",replay_buffer[index].qval)
    return replay_buffer




if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 50000
    MAX_EPISODE_STEPS = 1400
    TARGET_MODEL_UPDATE_INTERVAL = 50
    EPSILON_MIN = 0.05
    EPSILON_START = 0.5
    EPSLILON_COUNT = 6000 #Games
    INITIAL_RANDOM_STEPS = 5000
    RANDOM_GAME_EVERY = 20
    TRAIN_CRITIC_EVERY_N_STEP = 300
    CRITIC_TRAINING_SAMPLE_SIZE = 256
    TRAIN_ACTOR_EVERY_N_GAME = 1
    ACTOR_TRAINING_SAMPLE_SIZE = 8
    NUM_ACTOR_TRAINING_SAMPLES = 40
    TRAINING_ITTERATIONS = 1
    NUM_ACTOR_TRAINING_SAMPLES = 128
    PRINT_EVERY = 1
    RENDER_ENV = True
    LOAD_MODEL = True
    SAVE_MODEL = False
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 25

    epsilon = EPSILON_START
    env = gym.make('LunarLanderContinuous-v2')
    observation = env.reset()
    print("env action space ", env.action_space.shape)
    am = ActorModel(env.observation_space.shape,env.action_space.shape,lr=0.008)
    cm = CriticModel(env.observation_space.shape,env.action_space.shape,lr=0.01)
    agent = DQNAgent( am , cm )
    # import ipdb;ipdb.set_trace()

    if LOAD_MODEL:
        agent.actor_model = torch.load("actor"+MODEL_ID+MODEL_FILE_NAME)
        agent.critic_model = torch.load("critic"+MODEL_ID+MODEL_FILE_NAME)

        agent.actor_model.eval()
        agent.critic_model.eval()

    step_counter = 0
    last_step_count = 0


    action = []
    for game in range (NUM_GAMES):
        episode_sars = []
        score = 0
        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()

            action = agent.get_actions(observation).cpu().detach().numpy()

            observation_next, reward, done, info = env.step(action)
            score += reward

            observation = observation_next
            step_counter+=1
            last_step_count = step
            if done:

                break

        observation = env.reset()
        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        if (game%PRINT_EVERY==0):
            print("episide ", game,"last score",reward, "game score ", score ,"episode_len",last_step_count, "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
