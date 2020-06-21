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

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions,lr):
        super(Model,self).__init__()
        assert len(obs_shape) ==1, "This network only works on flat observations"
        self.obs_shape = obs_shape
        self.num_action = num_actions

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0],128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,num_actions)
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


def train_step(model, state_transitions, tgt, num_actions):
    if len(state_transitions) <=0:
        print("empty state transitions")
        return
    cur_states = torch.stack( ([torch.Tensor(s.state) for s in state_transitions]) ).to(model.device)
    rewards = torch.stack( ([torch.Tensor([s.reward]) for s in state_transitions]) ).to(model.device)
    Qs = torch.stack( ([torch.Tensor([s.qval]) for s in state_transitions]) ).to(model.device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(model.device)
    next_states = torch.stack( ([torch.Tensor(s.next_state) for s in state_transitions]) ).to(model.device)
    actions = [s.action for s in state_transitions]
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        # actual_Q_values = Qs
        pred_qvals_next = model(next_states).max(-1)[0]
    model.opt.zero_grad()
    pred_qvals = model(cur_states)

    one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions).to(model.device)
    # loss = torch.mean(torch.sqrt((torch.sum(pred_qvals*one_hot_actions,-1) - actual_Q_values.view(-1) )**2)).to(model.device)
    # loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), actual_Q_values.view(-1) )
    loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), rewards.view(-1)+0.99*mask[:,0]*pred_qvals_next.view(-1) ).mean()
    loss.backward()
    model.opt.step()
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
            replay_buffer[index].qval = replay_buffer[index].reward + 0.99 * replay_buffer[next_index].qval
            if(step_counter%2000==0):
                print("i",i,"q ",replay_buffer[index].qval)
    return replay_buffer


if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 50000
    MAX_EPISODE_STEPS = 1490
    TARGET_MODEL_UPDATE_INTERVAL = 50
    EPSILON_MIN = 0.05
    EPSILON_START = 0.5
    EPSLILON_COUNT = 6000 #Games
    RANDOM_GAME_EVERY = 20
    TRAIN_EVERY_N_STEPS = 25
    TRAINING_SAMPLE_SIZE = 256
    TRAINING_ITTERATIONS = 1
    PRINT_EVERY = 1
    RENDER_ENV = True
    LOAD_MODEL = True
    SAVE_MODEL = False
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 25

    epsilon = EPSILON_START
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v1')

    observation = env.reset()
    agent = DQNAgent(Model(env.observation_space.shape,env.action_space.n,lr=0.0001), Model(env.observation_space.shape,env.action_space.n,lr=0.0001) )
    if LOAD_MODEL:
        print("Loading Model ", ""+MODEL_ID+MODEL_FILE_NAME)
        agent.model = torch.load(""+MODEL_ID+MODEL_FILE_NAME)
        agent.model.eval()
    step_counter = 0
    avg_reward = []
    last_step_count = 0
    # qeval = m(torch.Tensor(allObs))
    # # print("allObs ", allObs)
    # # print("qeval ",qeval)


    for game in range (NUM_GAMES):
        # if game == 8:
        #     print("rb ",rb.buffer)
        score = 0

        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()
            # import ipdb; ipdb.set_trace()
            action = 0
            action = agent.get_actions(observation).item()


            observation_next, reward, done, info = env.step(action)

            score += reward

            observation = observation_next
            step_counter+=1
            last_step_count = step
            if done:

                observation = env.reset()
                break


        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        if (game%PRINT_EVERY==0):
            print("episide ", game,"last score",reward, "game score ", score ,"episode_len",last_step_count, "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
