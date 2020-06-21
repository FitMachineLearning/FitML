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
from PIL import Image


@dataclass
class sars:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
    qval: float

class DQNAgent:
    def __init__(self,model,targetModel):
        self.model = model
        self.targetModel = targetModel

    def get_actions(self, observations):
        q_vals = self.model(torch.Tensor(observations).to(self.model.device))
        return q_vals.max(-1)[1]

    def update_target_model(self):
        self.targetModel.load_state_dict(self.model.state_dict())

    def process_frame(self,frame):
        img = Image.fromarray(frame, 'RGB')
        width, height = img.size
        frame =  img.crop((5,35,width-15,height-15))
        return frame

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions,lr):
        super(Model,self).__init__()
        # assert len(obs_shape) ==1, "This network only works on flat observations"
        self.obs_shape = obs_shape
        self.num_action = num_actions
        # import ipdb; ipdb.set_trace()

        self.conv_net = torch.nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 8, 4),
            # nn.MaxPool2d(4),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3,1),
            # nn.MaxPool2d(4),
            # nn.Dropout(0.2),
            nn.ReLU()

        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(50176,256),
            # nn.Dropout(0.6),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128,256),
            # nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(256,num_actions)
        )
        self.opt = optim.Adam(self.conv_net.parameters(),lr=lr)
        self.opt2 = optim.Adam(self.linear_layer.parameters(),lr=lr)

        if torch.cuda.is_available():
            print("Using CUDA")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)


    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0),-1)
        x = self.linear_layer(x)
        return x



class ReplayBuffer:
    def __init__(self, buffer_size = 1000):
        # self.buffer_size = buffer_size
        self.buffer_size = buffer_size
        self.buffer = np.empty((buffer_size),dtype=object)

        # self.buffer = []
        self.index = 0

    def insert(self, sars):

        # if self.index>1000:
        #     # import ipdb; ipdb.set_trace()
        #     Qs = np.array([s.qval for s in self.buffer[0:(min(self.index,self.buffer_size))]])
        #     Qs_threshold = Qs.mean() + Qs.var()/4
        #     select_prob = 1 - ( ( sars.qval - Qs_threshold ) / Qs_threshold)
        #     select_prob = max(0.15,select_prob)
        #     if random()<select_prob:
        #         return



        self.buffer[self.index%self.buffer_size] = sars
        self.index+=1

    def sample(self, num_samples,current_episode_steps):
        # assert num_samples < min(len(self.buffer),self.index)
        # if num_samples>self.index:
        # print("sampling n ",min(num_samples,self.index))
        a = self.buffer[0:min(self.index,self.buffer_size)]
        if len(self.buffer) > 0:
            return np.random.choice(a, min(num_samples,self.index))
        else:
            return []

    def sample_top(self, num_samples,current_episode_steps):
        import ipdb; ipdb.set_trace()
        Qs = np.array([s.qvals for s in self.buffer])
        Qs_threshold = Qs.mean() + Qs.var()/3

        # if num_samples>self.index:
        # print("sampling n ",min(num_samples,self.index))
        a = self.buffer[0:min(self.index,self.buffer_size)]
        if len(self.buffer) > 0:
            return np.random.choice(a, min(num_samples,self.index))
        else:
            return []
