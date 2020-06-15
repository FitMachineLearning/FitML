import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, 64)
        self.dropout1 = nn.Dropout(p=0.1)
        self.affine2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.affine3 = nn.Linear(64, 4)

        self.saved_log_probs = []
        self.rewards = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)


    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout1(x)
        x = self.affine2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)
