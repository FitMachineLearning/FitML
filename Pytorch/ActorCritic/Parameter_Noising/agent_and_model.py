## DQN Tutorial
## Implementation from https://github.com/FitMachineLearning
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
    advantage: float = 0.0

class DQNAgent:
    def __init__(self,actor_model,critic_model):
        self.actor_model = actor_model
        self.critic_model = critic_model

    def get_actions(self, observations):
        # import ipdb; ipdb.set_trace()
        guessed_actions = self.actor_model(torch.Tensor(observations).to(self.actor_model.device))
        return guessed_actions

    def get_predicted_Q_values(self,observation_and_action):
        guessed_Qs = self.critic_model(torch.Tensor(observation_and_action))
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
            torch.nn.Linear(obs_shape[0],1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,action_shape[0])
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
            torch.nn.Linear(obs_shape[0]+action_shape[0],1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1) # one out put because we are predicting Q values
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
        # self.buffer_size = buffer_size
        self.buffer_size = buffer_size
        self.buffer = np.empty((buffer_size),dtype=object)

        # self.buffer = []
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
        self.buffer[self.index%self.buffer_size] = sars
        self.index+=1
        # self.buffer.append(sars)
        # if(len(self.buffer)>self.buffer_size):
        #     self.buffer = self.buffer[1:]
        #     # print("Clipping Buffer at size", len(self.buffer))

    def sample(self, num_samples,current_episode_steps):
        # assert num_samples < min(len(self.buffer),self.index)
        # if num_samples>self.index:
        # print("sampling n ",min(num_samples,self.index))
        a = self.buffer[0:min(self.index,self.buffer_size)]
        if len(self.buffer) > 0:
            return np.random.choice(a, min(num_samples,self.index))
        else:
            return []
