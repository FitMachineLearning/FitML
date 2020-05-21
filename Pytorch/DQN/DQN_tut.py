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
        # print("Observation ", observations)


        q_vals = self.model(torch.Tensor(observations))
        # print("q_vals", q_vals)
        # q_vals = q_vals.argmax(-1)
        # print("q_vals argmax", q_vals)
        # print("q_vals.argmax", q_vals)
        # retval = q_vals

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
            torch.nn.Linear(obs_shape[0],64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,num_actions)
        )
        self.opt = optim.Adam(self.net.parameters(),lr=lr)

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    def __init__(self, buffer_size = 1000):
        self.buffer_size = buffer_size
        self.buffer = []

    def insert(self, sars):
        self.buffer.append(sars)
        self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, num_samples):
        assert num_samples < len(self.buffer)
        return np.random.choice(self.buffer, num_samples)



def get_one_hot(action,n_dim):
    retval = np.zeros(n_dim)
    retval[action] = 1.0
    return retval


def train_step(model, state_transitions, tgt, num_actions):
    cur_states = torch.stack( ([torch.Tensor(s.state) for s in state_transitions]) )
    rewards = torch.stack( ([torch.Tensor([s.reward]) for s in state_transitions]) )
    Qs = torch.stack( ([torch.Tensor([s.qval]) for s in state_transitions]) )
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack( ([torch.Tensor(s.next_state) for s in state_transitions]) )
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        pred_qvals_next = Qs
        # pred_qvals_next = tgt(next_states)
        # pred_qvals_next = pred_qvals_next.max(axis=1)[0]

    model.opt.zero_grad()
    pred_qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions)

    # loss = (rewards + mask[:,0]*pred_qvals_next - torch.sum(pred_qvals*one_hot_actions,-1)).mean()
    loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), (rewards + mask[:,0]*pred_qvals_next) )
    loss.backward()
    model.opt.step()
    # print("loss ", loss)
    return loss

def update_Qs(replay_buffer,step_counter,episode_len,buffer_size):
    for i in range(episode_len):
        index = ((step_counter-1)-i) %buffer_size
        next_index = (step_counter-i)%buffer_size
        if i==0:
            replay_buffer[index].qval = replay_buffer[index].reward
            # print("i",i,"q ",replay_buffer[index].qval)
        else:
            replay_buffer[index].qval = replay_buffer[index].reward + 0.98 * replay_buffer[next_index].qval
            # print("i",i,"q ",replay_buffer[index].qval)


def train_step2(model,state_transitions,targetModel,num_actions):
    # print("state_transitions" , state_transitions)
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions]))
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))

    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions]))
    # act = torch.Tensor(np.zeros(num_actions))
    actions = torch.stack([torch.Tensor(get_one_hot(action,num_actions)) for s in state_transitions])

    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])

    with torch.no_grad():
        # qevals_next = targetModel(next_states).max(-1)
        qevals_next = targetModel(next_states)
        # print("qevals_next",qevals_next)
        qevals_next = qevals_next.max(axis=1)[0]
        # print("qevals_next . max",qevals_next)

    model.opt.zero_grad()
    qevals = model(cur_states)

    # print("rewards ",rewards.shape, rewards)
    # print("qevals ",qevals.shape,qevals)
    # # print("maks ",mask.shape,mask)
    # print("actions ",actions.shape,actions)
    print("qevals_next",qevals_next)
    #
    print("qeval*actions ",  torch.sum(qevals*actions,axis=1) )
    # print("qeval*actions . mean() ",  torch.sum(qevals*actions,axis=1).mean() )


    loss =  ( (rewards + 0.98 * qevals_next*mask[:,0] ) - (torch.sum(qevals*actions,axis=1)) ).mean()
    # loss =  ( (rewards + 0.98 * qevals_next*mask) - qevals*actions ).mean()
    loss.backward()
    model.opt.step()

    print("Loss ", loss)
    return loss


if __name__=='__main__':
    NUM_GAMES = 50000
    MAX_EPISODE_STEPS = 600
    TARGET_MODEL_UPDATE_INTERVAL = 50
    EPSILON_MIN = 0.01
    EPSILON_START = 0.8
    EPSLILON_COUNT = 300 #Games

    epsilon = EPSILON_START
    # env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')

    observation = env.reset()
    # obs2 = np.random.random(4)
    # allObs = np.array([observation,obs2])
    m = Model(env.observation_space.shape,env.action_space.n,lr=0.001)
    rb = ReplayBuffer(100000)
    agent = DQNAgent(m, Model(env.observation_space.shape,env.action_space.n,lr=0.001) )
    step_counter = 0
    # qeval = m(torch.Tensor(allObs))
    # # print("allObs ", allObs)
    # # print("qeval ",qeval)


    for game in range (NUM_GAMES):
        if game == 8:
            print("rb ",rb.buffer)
        if game%TARGET_MODEL_UPDATE_INTERVAL == 0 :
            print("game", game," updating target model")
            agent.update_target_model()
        for step in range (MAX_EPISODE_STEPS):
            env.render()
            # import ipdb; ipdb.set_trace()
            action = 0
            if step_counter<1000 or random()<epsilon or game%10==0:
                action = env.action_space.sample()
                # print("random action")
            else:
                action = agent.get_actions(observation).item()
                # print("*** action ",action)
                # action = action.data.cpu()

            observation_next, reward, done, info = env.step(action)
            if done:
                reward=-100
            _sars = sars(observation,action,reward,observation_next,done,0.0)
            rb.insert(_sars)
            observation = observation_next
            if len(rb.buffer) > 1000 and step_counter%3==0:
                train_step(agent.model,rb.sample(2),agent.targetModel,env.action_space.n)
                # print("rb  size ", len(rb.buffer))

            step_counter+=1
            if done:

                # print("last reward ", reward)
                observation = env.reset()
                update_Qs(rb.buffer,step_counter,step,rb.buffer_size)
                break
        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        print("epsilon ", epsilon)
