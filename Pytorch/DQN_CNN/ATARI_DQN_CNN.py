## DQN Tutorial
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
from agent_and_model import DQNAgent,sars, Model, ReplayBuffer
import plotly.graph_objects as go


def get_one_hot(action,n_dim):
    retval = np.zeros(n_dim)
    retval[action] = 1.0
    return retval


def train_step(model, state_transitions, tgt, num_actions, gamma):
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
        actual_Q_values = Qs
        # import ipdb; ipdb.set_trace()
        pred_qvals_next = model(next_states.view(len(state_transitions),3,160,140*3)).max(-1)[0]
    model.opt.zero_grad()
    pred_qvals = model(cur_states.view(len(state_transitions),3,160,140*3))

    one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions).to(model.device)
    # loss = torch.mean(torch.sqrt((torch.sum(pred_qvals*one_hot_actions,-1) - actual_Q_values.view(-1) )**2)).to(model.device)
    loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), actual_Q_values.view(-1) )
    # loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), rewards.view(-1)+gamma*mask[:,0]*pred_qvals_next.view(-1) ).mean()
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
            # if(step_counter%2000==0):
            #     # print("i",i,"q ",replay_buffer[index].qval)
        else:
            replay_buffer[index].qval = replay_buffer[index].reward + 0.98 * replay_buffer[next_index].qval
            # if(step_counter%2000==0):
            #     # print("i",i,"q ",replay_buffer[index].qval)
    return replay_buffer

def plot_score(all_scores):
    fig = go.Figure(data=go.Bar(y=all_scores))
    fig.write_html('DQN_CNN_Trend_figure.html')

if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 50000
    INITIAL_RANDOM_STEPS = 20000
    MAX_EPISODE_STEPS = 4000
    LEARNING_RATE = 0.0001
    TARGET_MODEL_UPDATE_INTERVAL = 50
    GAMMA_DISCOUNT_FACTOR = 0.98
    EPSILON_MIN = 0.15
    EPSILON_START = 0.6
    EPSLILON_COUNT = 500 #Games
    RANDOM_GAME_EVERY = 5
    TRAIN_EVERY_N_STEPS = 15
    TRAINING_SAMPLE_SIZE = 128
    TRAINING_ITTERATIONS = 1
    PRINT_EVERY = 1
    RENDER_ENV = False
    LOAD_MODEL = False
    SAVE_MODEL = True
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 5

    epsilon = EPSILON_START
    env = gym.make('Pong-v0')
    # env = gym.make('CartPole-v1')

    observation = env.reset()


    # observation = env.render(mode='rgb_array')

    # obs2 = np.random.random(4)
    # allObs = np.array([observation,obs2])



    rb = ReplayBuffer(10000)
    agent = DQNAgent(Model(env.observation_space.shape,env.action_space.n,lr=LEARNING_RATE), Model(env.observation_space.shape,env.action_space.n,lr=LEARNING_RATE) )
    noisy_agent = DQNAgent(Model(env.observation_space.shape,env.action_space.n,lr=LEARNING_RATE), Model(env.observation_space.shape,env.action_space.n,lr=LEARNING_RATE) )

    frame1 = []
    frame2 = []
    frame3 = []
    frame1 = agent.process_frame(observation)
    frame2 = agent.process_frame(observation)
    frame3 = agent.process_frame(observation)
    # import ipdb; ipdb.set_trace()
    observation = np.concatenate((frame1,frame2,frame3),axis=1)
    observation = observation.reshape((1,3,160,140*3))
    if LOAD_MODEL:
        agent.model.load_state_dict(torch.load(""+MODEL_ID+MODEL_FILE_NAME))
        agent.model.eval()
    step_counter = 0
    avg_reward = []
    # qeval = m(torch.Tensor(allObs))
    # # print("allObs ", allObs)
    # # print("qeval ",qeval)
    all_scores = []


    for game in range (NUM_GAMES):
        # if game == 8:
        #     print("rb ",rb.buffer)
        episode_sars = []
        score = 0.0
        # if game%NOISY_AGENT_GAME_EVERY==0:
        #     print("adding param noise")
        #     #copy the policy model
        #     noisy_agent.actor_model.load_state_dict(agent.actor_model.state_dict())
        #     #add nois to the copy of the policy model
        #     with torch.no_grad():
        #         for param in noisy_agent.actor_model.parameters():
        #             param.add_(torch.randn(param.size()).to(noisy_agent.actor_model.device) * 0.02)

        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()
            # import ipdb; ipdb.set_trace()



            action = 0
            if step_counter<INITIAL_RANDOM_STEPS or random()<epsilon or game%RANDOM_GAME_EVERY==0:
                action = env.action_space.sample()
                # print("random action")
            else:
                # import ipdb; ipdb.set_trace()
                # print("obs ", observation.shape)
                action = agent.get_actions(observation).item()


            frame3 = frame2
            frame2 = frame1
            frame1, reward, done, info = env.step(action)


            score += reward
            # print("frame1", frame1.shape)
            frame1 = agent.process_frame(frame1)
            observation_next = np.concatenate((frame1,frame2,frame3),axis=1)

            # print("obs - concatenate", observation_next.shape)
            # if True or step%100==99:
            #     img = Image.fromarray(observation_next, 'RGB')
            #     img.save('my.png')
            #     img.show()
            # if done and reward <=-100:
            #     reward = -300
            observation_next = observation_next.reshape((1,3,160,140*3))
            # print("obs - concatenate - after ", observation_next.shape)
            # observation_next = obs

            _sars = sars(observation,action,reward,observation_next,done,0.0)
            # reward *= 100
            episode_sars.append(_sars)
            avg_reward.append([reward])
            # if(reward==-100):
            #     print("Adding -100 ",reward)
            if rb.index > INITIAL_RANDOM_STEPS and step_counter%TRAIN_EVERY_N_STEPS==0:
                # import ipdb; ipdb.set_trace()
                # print("Training Agent ")

                for s in range(TRAINING_ITTERATIONS):
                    dick = rb.sample(TRAINING_SAMPLE_SIZE,step)
                    train_step(agent.model,dick,agent.targetModel,env.action_space.n, GAMMA_DISCOUNT_FACTOR)
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
                frame1 = agent.process_frame(observation)
                frame2 = agent.process_frame(observation)
                frame3 = agent.process_frame(observation)

                observation = np.concatenate((frame1,frame2,frame3),axis=1)
                observation = observation.reshape((1,3,160,140*3))
                break

        all_scores.append(score)
        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        if (game%PRINT_EVERY==0):
            plot_score(all_scores)
            print("episide ", game,"game score",score ,"episode_len", len(episode_sars),"buffer",min(rb.index,rb.buffer_size), "score", np.average( avg_reward), "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
