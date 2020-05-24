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
            torch.nn.Linear(obs_shape[0],32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,num_actions)
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
    with torch.no_grad():
        actual_Q_values = Qs
    model.opt.zero_grad()
    pred_qvals = model(cur_states)

    one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions).to(model.device)
    # loss = torch.mean(torch.sqrt((torch.sum(pred_qvals*one_hot_actions,-1) - actual_Q_values.view(-1) )**2)).to(model.device)
    loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), actual_Q_values.view(-1) )
    loss.backward()
    model.opt.step()
    return loss

# def train_step3(model, state_transitions, tgt, num_actions):
#     cur_states = torch.stack( ([torch.Tensor(s.state) for s in state_transitions]) ).to(model.device)
#     rewards = torch.stack( ([torch.Tensor([s.reward]) for s in state_transitions]) ).to(model.device)
#     Qs = torch.stack( ([torch.Tensor([s.qval]) for s in state_transitions]) ).to(model.device)
#     mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(model.device)
#     next_states = torch.stack( ([torch.Tensor(s.next_state) for s in state_transitions]) ).to(model.device)
#     actions = [s.action for s in state_transitions]
#     with torch.no_grad():
#         qvals_next = tgt(next_states).max(-1)[0]
#     model.opt.zero_grad()
#     qvals = model(cur_states)
#     one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions).to(model.device)
#     loss = (
#             ( rewards + mask[:,0]*qvals_next - torch.sum(qvals*one_hot_actions,-1) )
#         ).mean()
#     loss.backward()
#     model.opt.step()
#     return loss



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

# def train_step2(model,state_transitions,targetModel,num_actions):
#     # print("state_transitions" , state_transitions)
#     cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions]))
#     next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))
#
#     rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions]))
#     # act = torch.Tensor(np.zeros(num_actions))
#     actions = torch.stack([torch.Tensor(get_one_hot(action,num_actions)) for s in state_transitions])
#
#     mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])
#
#     with torch.no_grad():
#         # qevals_next = targetModel(next_states).max(-1)
#         qevals_next = targetModel(next_states)
#         # print("qevals_next",qevals_next)
#         qevals_next = qevals_next.max(axis=1)[0]
#         # print("qevals_next . max",qevals_next)
#
#     model.opt.zero_grad()
#     qevals = model(cur_states)
#
#     # print("rewards ",rewards.shape, rewards)
#     # print("qevals ",qevals.shape,qevals)
#     # # print("maks ",mask.shape,mask)
#     # print("actions ",actions.shape,actions)
#     print("qevals_next",qevals_next)
#     #
#     print("qeval*actions ",  torch.sum(qevals*actions,axis=1) )
#     # print("qeval*actions . mean() ",  torch.sum(qevals*actions,axis=1).mean() )
#
#
#     loss =  ( (rewards + 0.98 * qevals_next*mask[:,0] ) - (torch.sum(qevals*actions,axis=1)) ).mean()
#     # loss =  ( (rewards + 0.98 * qevals_next*mask) - qevals*actions ).mean()
#     loss.backward()
#     model.opt.step()
#
#     print("Loss ", loss)
#     return loss


if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 50000
    MAX_EPISODE_STEPS = 1000
    TARGET_MODEL_UPDATE_INTERVAL = 50
    EPSILON_MIN = 0.05
    EPSILON_START = 0.5
    EPSLILON_COUNT = 6000 #Games
    RANDOM_GAME_EVERY = 20
    TRAIN_EVERY_N_STEPS = 25
    TRAINING_SAMPLE_SIZE = 256
    TRAINING_ITTERATIONS = 1
    PRINT_EVERY = 2
    RENDER_ENV = False
    LOAD_MODEL = False
    SAVE_MODEL = True
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 25

    epsilon = EPSILON_START
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v1')

    observation = env.reset()
    # obs2 = np.random.random(4)
    # allObs = np.array([observation,obs2])



    rb = ReplayBuffer(30000)
    agent = DQNAgent(Model(env.observation_space.shape,env.action_space.n,lr=0.01), Model(env.observation_space.shape,env.action_space.n,lr=0.01) )
    if LOAD_MODEL:
        agent.model.load_state_dict(torch.load(""+MODEL_ID+MODEL_FILE_NAME))
        agent.model.eval()
    step_counter = 0
    avg_reward = []
    # qeval = m(torch.Tensor(allObs))
    # # print("allObs ", allObs)
    # # print("qeval ",qeval)


    for game in range (NUM_GAMES):
        # if game == 8:
        #     print("rb ",rb.buffer)
        episode_sars = []
        if game%TARGET_MODEL_UPDATE_INTERVAL == 0 :
            print("game", game," updating target model")
            agent.update_target_model()
        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()
            # import ipdb; ipdb.set_trace()
            action = 0
            if step_counter<2000 or random()<epsilon or game%RANDOM_GAME_EVERY==0:
                action = env.action_space.sample()
                # print("random action")
            else:
                action = agent.get_actions(observation).item()
                # print("*** action ",action)
                # action = action.data.cpu()

            observation_next, reward, done, info = env.step(action)
            # if done and reward <=-100:
            #     reward = -300
            _sars = sars(observation,action,reward,observation_next,done,0.0)
            episode_sars.append(_sars)
            avg_reward.append([reward])
            # if(reward==-100):
            #     print("Adding -100 ",reward)
            if rb.index > 3000 and step_counter%TRAIN_EVERY_N_STEPS==0:
                # import ipdb; ipdb.set_trace()

                for s in range(TRAINING_ITTERATIONS):
                    dick = rb.sample(TRAINING_SAMPLE_SIZE,step)
                    train_step(agent.model,dick,agent.targetModel,env.action_space.n)
                    # print("training  size ",rb.index%rb.buffer_size, " - sample ",dick)
            observation = observation_next
            step_counter+=1
            if done:


                # print("last reward ", reward)

                episode_sars = update_Qs(episode_sars,step_counter,step,len(episode_sars))
                for j in range(len(episode_sars)):
                    rb.insert(episode_sars[j])

                if(SAVE_MODEL and game%SAVE_MODEL_EVERY==0 and game>50):
                    # torch.save(agent.model.state_dict(),""+MODEL_ID+MODEL_FILE_NAME)
                    torch.save(agent.model,""+MODEL_ID+MODEL_FILE_NAME)


                observation = env.reset()
                break


        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        if (game%PRINT_EVERY==0):
            print("episide ", game,"last score",reward ,"episode_len", len(episode_sars),"buffer",len(rb.buffer), "score", np.average( avg_reward), "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
