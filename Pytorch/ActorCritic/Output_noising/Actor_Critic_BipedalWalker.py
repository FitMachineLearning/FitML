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
    state_action: Any
    next_state_action : Any

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
            torch.nn.Linear(obs_shape[0],32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,action_shape[0])
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
            torch.nn.Linear(obs_shape[0]+action_shape[0],64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1) # one out put because we are predicting Q values
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
        self.buffer = [None]*buffer_size
        # self.buffer = []
        self.index = 0

    def insert(self, sars):
        # self.buffer.append(sars)
        # print("inserting index ", self.index, "@",self.index%self.buffer_size)
        if(self.index == 10):
            print("first 10 ",self.buffer[0:10])
            # import ipdb; ipdb.set_trace()

        current_index = self.index%self.buffer_size
        self.buffer[current_index] = sars
        self.index+=1

        # self.buffer.append(sars)
        # if(len(self.buffer)>self.buffer_size):
        #     self.buffer = self.buffer[1:]
            # print("Clipping Buffer at size", len(self.buffer))

    def sample(self, num_samples,current_episode_steps):
        # assert num_samples < min(len(self.buffer),self.index)
        # if num_samples>self.index:
        # print("sampling n ",min(num_samples,self.index))
        max_index = 0
        if self.index < self.buffer_size:
            max_index = self.index
        else:
            max_index = self.buffer_size
        a = np.random.choice(self.buffer[0:max_index],num_samples)
        return a

        # if len(self.buffer) > 0:
        #     return np.random.choice(self.buffer, min(num_samples,self.index))
        # else:
        #     return []



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
            # add random gaussian noise from predicted actions
            for j in range(num_actor_training_samples):
                noise = np.random.normal(0,0.15,num_actions)
                random_actions.append(act+noise)



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
        best_state_action.append(sars(random_states[i_states], random_actions[np.argmax(QAs)],0.0,None,False,np.max(QAs),None,None ))
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

def train_critic(critic_model, state_transitions, num_actions,gamma):
    if len(state_transitions) <=0:
        print("empty state transitions")
        return


    cur_states = torch.stack( ([torch.Tensor(torch.cat((torch.Tensor(s.state),torch.Tensor(s.action)),0)) for s in state_transitions]) ).to(critic_model.device)


    rewards = torch.stack( ([torch.Tensor([s.reward]) for s in state_transitions]) ).to(critic_model.device)
    Qs = torch.stack( ([torch.Tensor([s.qval]) for s in state_transitions]) ).to(critic_model.device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(critic_model.device)
    next_states = torch.stack( ([torch.Tensor(s.next_state) for s in state_transitions]) ).to(critic_model.device)
    # for i in range(len(state_transitions)):
    #     print(" ", i, " ", state_transitions[i].next_state_action)
    # import ipdb; ipdb.set_trace()
    next_state_action = torch.stack( ([torch.Tensor(s.next_state_action) for s in state_transitions]) ).to(critic_model.device)
    actions = [s.action for s in state_transitions]
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        actual_Q_values = Qs
        # pred_qvals_next = critic_model(next_state_action)[0]
    critic_model.opt.zero_grad()
    pred_qvals = critic_model(cur_states)

    # one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions).to(model.device)
    # loss = torch.mean(torch.sqrt((torch.sum(pred_qvals*one_hot_actions,-1) - actual_Q_values.view(-1) )**2)).to(model.device)
    loss = F.smooth_l1_loss(pred_qvals.view(-1), actual_Q_values.view(-1) )
    # loss = F.smooth_l1_loss(torch.sum(pred_qvals,-1), rewards.view(-1)+gamma*mask[:,0]*pred_qvals_next.view(-1) ).mean()
    loss.backward()
    critic_model.opt.step()
    return loss

def update_Qs(replay_buffer,step_counter,episode_len,buffer_size,gamma):
    for i in range(episode_len+1):
        # if(step_counter > buffer_size):
        # import ipdb; ipdb.set_trace()
        index = episode_len-1-i
        next_index = index+1
        if i==0:
            replay_buffer[index].qval = replay_buffer[index].reward
            replay_buffer[index].next_state_action = replay_buffer[index].state_action

            if(step_counter%2000==0):
                print("i",i,"q ",replay_buffer[index].next_state_action)
        else:
            replay_buffer[index].qval = replay_buffer[index].reward + gamma * replay_buffer[next_index].qval
            replay_buffer[index].next_state_action = replay_buffer[next_index].state_action
            if(step_counter%2000==0):
                print("i",i,"q ",replay_buffer[index].next_state_action)
    return replay_buffer




if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 50000
    MAX_EPISODE_STEPS = 400
    TARGET_MODEL_UPDATE_INTERVAL = 50
    GAMMA_DISCOUNT_FACTOR = 0.995
    EPSILON_MIN = 0.05
    EPSILON_START = 0.5
    EPSLILON_COUNT = 3000 #Games
    INITIAL_RANDOM_STEPS = 2000
    RANDOM_GAME_EVERY = 20
    TRAIN_CRITIC_EVERY_N_STEP = 3
    CRITIC_TRAINING_SAMPLE_SIZE = 256
    TRAIN_ACTOR_EVERY_N_STEPS = 25*2
    ACTOR_TRAINING_SAMPLE_SIZE = 4
    NUM_ACTOR_TRAINING_SAMPLES = 20
    TRAINING_ITTERATIONS = 1
    NUM_ACTOR_TRAINING_SAMPLES = 128
    PRINT_EVERY = 1
    RENDER_ENV = False
    LOAD_MODEL = False
    SAVE_MODEL = True
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 25

    epsilon = EPSILON_START
    env = gym.make('BipedalWalker-v3')
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('CartPole-v1')

    observation = env.reset()
    # obs2 = np.random.random(4)
    # allObs = np.array([observation,obs2])


    # import ipdb;ipdb.set_trace()
    rb = ReplayBuffer(50000)
    print("env action space ", env.action_space.shape)
    am = ActorModel(env.observation_space.shape,env.action_space.shape,lr=0.00101)
    cm = CriticModel(env.observation_space.shape,env.action_space.shape,lr=0.001)
    agent = DQNAgent( am , cm )
    # import ipdb;ipdb.set_trace()

    if LOAD_MODEL:
        agent.actor_model.load_state_dict(torch.load("actor"+MODEL_ID+MODEL_FILE_NAME))
        agent.critic_model.load_state_dict(torch.load("critic"+MODEL_ID+MODEL_FILE_NAME))

        agent.actor_model.eval()
        agent.critic_model.eval()

    step_counter = 0
    avg_reward = []
    # qeval = m(torch.Tensor(allObs))
    # # print("allObs ", allObs)
    # # print("qeval ",qeval)

    action = []
    for game in range (NUM_GAMES):
        # if game == 8:
        #     print("rb ",rb.buffer)
        episode_sars = []
        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()
            # import ipdb; ipdb.set_trace()
            action = []
            if step_counter<INITIAL_RANDOM_STEPS or random()<epsilon or game%RANDOM_GAME_EVERY==0:
                action = env.action_space.sample()
                # print("random action")
            else:
                # import ipdb; ipdb.set_trace()
                action = agent.get_actions(observation).cpu().detach().numpy()

            observation_next, reward, done, info = env.step(action)
            # if done and reward <=-100:
            #     reward = -300
            reward *= 10
            _sars = sars(observation,action,reward,observation_next,done,0.0,np.concatenate((observation,action)),None)
            episode_sars.append(_sars)
            avg_reward.append([reward])
            # if(reward==-100):
            #     print("Adding -100 ",reward)
            if rb.index > INITIAL_RANDOM_STEPS and step_counter%TRAIN_CRITIC_EVERY_N_STEP==0:
                # import ipdb; ipdb.set_trace()
                if step_counter%1000==0:
                    print("Training critic.")
                for s in range(TRAINING_ITTERATIONS):
                    samples = rb.sample(CRITIC_TRAINING_SAMPLE_SIZE,step)
                    train_critic(agent.critic_model, samples, env.action_space.shape[0],GAMMA_DISCOUNT_FACTOR)

            if rb.index > INITIAL_RANDOM_STEPS*2 and step_counter%TRAIN_ACTOR_EVERY_N_STEPS==0 :
                samples = rb.sample(ACTOR_TRAINING_SAMPLE_SIZE,0)
                if step_counter%1000==0:
                    print("Training actor")
                train_actor(agent.actor_model, agent.critic_model, samples, NUM_ACTOR_TRAINING_SAMPLES, env.action_space.shape[0])


                    # print("training  size ",rb.index%rb.buffer_size, " - sample ",dick)
            observation = observation_next
            step_counter+=1
            if done:


                # print("last reward ", reward)



                if(SAVE_MODEL and game%SAVE_MODEL_EVERY==0 and game>50):
                    # torch.save(agent.model.state_dict(),""+MODEL_ID+MODEL_FILE_NAME)
                    torch.save(agent.actor_model,"actor"+MODEL_ID+MODEL_FILE_NAME)
                    torch.save(agent.critic_model,"critic"+MODEL_ID+MODEL_FILE_NAME)


                observation = env.reset()
                break

        episode_sars = update_Qs(episode_sars,step_counter,len(episode_sars),len(episode_sars),GAMMA_DISCOUNT_FACTOR)
        for j in range(len(episode_sars)):
            rb.insert(episode_sars[j])

        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        if (game%PRINT_EVERY==0):
            print("episide ", game,"last score",reward ,"episode_len", len(episode_sars),"buffer",len(rb.buffer), "score", np.average( avg_reward), "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
        observation = env.reset()
