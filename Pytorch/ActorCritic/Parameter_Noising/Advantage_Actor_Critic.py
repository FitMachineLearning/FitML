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
from agent_and_model import sars,DQNAgent,ActorModel,CriticModel,ReplayBuffer
import plotly.graph_objects as go


def calc_advantage(agent , sars_recorded, action_predicted ):
    actual_q = sars_recorded.qval
    actor_predicted_action = action_predicted
    actor_state_action = np.concatenate((sars_recorded.state,actor_predicted_action),axis=0)
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        actor_predicted_q = agent.critic_model( torch.Tensor(   actor_state_action   ).to(agent.critic_model.device) ).cpu()
        # actor_predicted_q = agent.get_predicted_Q_values(actor_state_action)

    # import ipdb;ipdb.set_trace()
    advantage =  actual_q - actor_predicted_q
    return (advantage / abs(actor_predicted_q)).cpu().detach().numpy()[0] , actor_predicted_q.cpu().detach().numpy()[0]

def get_one_hot(action,n_dim):
    retval = np.zeros(n_dim)
    retval[action] = 1.0
    return retval

def train_actor_with_advantage(agent, state_transitions):
    random_states = [s.state for s in state_transitions]
    random_actions = []
    selected_states = []

    for i in range(len(state_transitions)):
        with torch.no_grad():

            # act = agent.actor_model(torch.Tensor(random_states[i]).to(actor_model.device)) .cpu().detach().numpy()
            # random_actions.append(act)
            predicted_action = agent.actor_model(torch.Tensor(random_states[i]).to(agent.actor_model.device)) .cpu().detach().numpy()
            advantage,actor_predicted_q = calc_advantage(agent,state_transitions[i],predicted_action)
            if advantage > 0 and random() < advantage:
                # print("adding for training advantage ", advantage, "actual q",state_transitions[i].qval, actor_predicted_q)
                state_transitions[i].advantage = advantage
                # print("state transitions adv ",advantage)
                selected_states.append(state_transitions[i])

    if len(selected_states)<=0:
        return 0
    else:
        print("selected num ",len(selected_states))
    t_advantages = torch.stack( ([torch.Tensor([s.advantage]) for s in selected_states]) ).to(agent.actor_model.device)
    t_random_states = torch.stack( ([torch.Tensor(s.state) for s in selected_states]) ).to(agent.actor_model.device)
    target_actions = torch.stack( ([torch.Tensor(s.action) for s in selected_states]) ).to(agent.actor_model.device)
    agent.actor_model.zero_grad()
    predicted_actions = agent.actor_model(t_random_states)

    # import ipdb; ipdb.set_trace()
    # loss = F.smooth_l1_loss(torch.sum(pred_qvals*one_hot_actions,-1), actual_Q_values.view(-1) )
    # loss = torch.mean(torch.sqrt((torch.sum(pred_qvals*one_hot_actions,-1) - actual_Q_values.view(-1) )**2)).to(model.device)
    # import ipdb; ipdb.set_trace()
    loss = F.l1_loss(predicted_actions, target_actions)
    # loss = - torch.sqrt(  torch.log(  torch.mul(   torch.sum( (predicted_actions - target_actions)**2,1 ),t_advantages)  )    )    .mean()
    # loss = torch.norm((predicted_actions - target_actions),2)*t_advantages
    loss.backward()
    agent.actor_model.opt.step()
    return loss

def train_actor(actor_model, critic_model,noisy_actor_model, state_transitions, num_actor_training_samples, num_actions):
    #for each observation get the critic to generate the Q value corresponding to each action_space
    #retain action observation pairs corresponding to the highest Q values
    #train the actor to converge towards that set

    #Generate random actions
    random_actions = []
    for i in range(num_actor_training_samples):
        random_actions.append( np.random.rand(num_actions)*2-1 )
        # import ipdb; ipdb.set_trace()
    #Get random observations
    for i in range(len(state_transitions)):
        random_actions.append(state_transitions[i].action)
    random_states = [s.state for s in state_transitions]

    # import ipdb; ipdb.set_trace()

    # for earch state add the best corresponding action to random actions
    for i in range(len(random_states)):
        with torch.no_grad():
            act = actor_model(torch.Tensor(random_states[i]).to(actor_model.device)) .cpu().detach().numpy()
            random_actions.append(act)
            act = noisy_actor_model(torch.Tensor(random_states[i]).to(noisy_actor_model.device)) .cpu().detach().numpy()
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
    # loss =  torch.sqrt(torch.sum( (predicted_actions - target_actions)**2,1 ) )    .mean()
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
    # import ipdb; ipdb.set_trace()
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


def plot_score(all_scores):
    fig = go.Figure(data=go.Bar(y=all_scores))
    fig.write_html('Trend_figure.html')

if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 80000
    MAX_EPISODE_STEPS = 400
    TARGET_MODEL_UPDATE_INTERVAL = 50
    EPSILON_MIN = 0.05
    EPSILON_START = 0.25
    EPSLILON_COUNT = 1000 #Games
    INITIAL_RANDOM_STEPS = 5000
    RANDOM_GAME_EVERY = 10
    NOISY_AGENT_GAME_EVERY = 3
    CRITIC_TRAINING_ITTERATIONS = 8
    TRAIN_CRITIC_EVERY_N_STEP = 15
    CRITIC_TRAINING_SAMPLE_SIZE = 1
    TRAIN_ACTOR_EVERY_N_STEP = 50
    TRAIN_ACTOR_EVERY_N_GAME = 1
    ACTOR_TRAINING_SAMPLE_SIZE = 1
    ACTOR_TRAINING_ITTERTIONS = 8
    LAST_EPISODE_TRAINING_SAMPLE_SIZE = 8
    # NUM_ACTOR_TRAINING_SAMPLES = 40
    # NUM_ACTOR_TRAINING_SAMPLES = 128
    PRINT_EVERY = 1
    RENDER_ENV = False
    LOAD_MODEL = False
    SAVE_MODEL = True
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 10

    epsilon = EPSILON_START
    # env = gym.make('BipedalWalker-v3')
    env = gym.make('LunarLanderContinuous-v2')

    # env = gym.make('CartPole-v1')

    observation = env.reset()
    # obs2 = np.random.random(4)
    # allObs = np.array([observation,obs2])


    # import ipdb;ipdb.set_trace()
    rb = ReplayBuffer(400000)
    print("env action space ", env.action_space.shape)
    am = ActorModel(env.observation_space.shape,env.action_space.shape,lr=0.000101)
    cm = CriticModel(env.observation_space.shape,env.action_space.shape,lr=0.0001)
    agent = DQNAgent( am , cm )
    n_am = ActorModel(env.observation_space.shape,env.action_space.shape,lr=0.008)
    n_cm = CriticModel(env.observation_space.shape,env.action_space.shape,lr=0.01)
    noisy_agent = DQNAgent(n_am, n_cm)
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
    all_scores = []
    for game in range (NUM_GAMES):
        # if game == 8:
        #     print("rb ",rb.buffer)
        score =0
        episode_sars = []
        if game%NOISY_AGENT_GAME_EVERY==0:
            print("adding param noise")
            #copy the policy model
            noisy_agent.actor_model.load_state_dict(agent.actor_model.state_dict())
            #add nois to the copy of the policy model
            with torch.no_grad():
                for param in noisy_agent.actor_model.parameters():
                    param.add_(torch.randn(param.size()).to(noisy_agent.actor_model.device) * 0.02)
        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()
            # import ipdb; ipdb.set_trace()
            action = []
            if step_counter<INITIAL_RANDOM_STEPS or random()<epsilon or game%RANDOM_GAME_EVERY==0:
                action = env.action_space.sample()
                # print("random action")
            elif step_counter>=INITIAL_RANDOM_STEPS and  game%NOISY_AGENT_GAME_EVERY ==0:
                if step%100==0:
                    print("noisy agent acting")
                action = noisy_agent.get_actions(observation).cpu().detach().numpy()
            else:
                # import ipdb; ipdb.set_trace()
                action = agent.get_actions(observation).cpu().detach().numpy()

            observation_next, reward, done, info = env.step(action)
            # reward = reward*100
            if step >= MAX_EPISODE_STEPS:
                done = True
            _sars = sars(observation,action,reward,observation_next,done,0.0)
            episode_sars.append(_sars)
            avg_reward.append([reward])
            score += reward
            # if(reward==-100):
            #     print("Adding -100 ",reward)
            if rb.index > INITIAL_RANDOM_STEPS and step_counter%TRAIN_CRITIC_EVERY_N_STEP==0:
                # import ipdb; ipdb.set_trace()
                # print("Training critic.")
                for s in range(CRITIC_TRAINING_ITTERATIONS):
                    samples = rb.sample(CRITIC_TRAINING_SAMPLE_SIZE,step)
                    train_critic(agent.critic_model, samples, env.action_space.shape[0])

            if rb.index > INITIAL_RANDOM_STEPS and step_counter%TRAIN_ACTOR_EVERY_N_STEP==0:
                for s in range(ACTOR_TRAINING_ITTERTIONS):
                    samples = rb.sample(ACTOR_TRAINING_SAMPLE_SIZE,0)
                    if rb.index > INITIAL_RANDOM_STEPS and game%TRAIN_ACTOR_EVERY_N_GAME==0 :
                        # train_actor_with_advantage(agent,  samples)
                        train_actor(agent.actor_model, agent.critic_model, noisy_agent.actor_model, samples, ACTOR_TRAINING_SAMPLE_SIZE, env.action_space.shape[0])

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
                    torch.save(agent.actor_model,"A2C_actor"+MODEL_ID+MODEL_FILE_NAME)
                    torch.save(agent.critic_model,"A2C_critic"+MODEL_ID+MODEL_FILE_NAME)


                break
        observation = env.reset()

        for s in range(ACTOR_TRAINING_ITTERTIONS):
            samples = rb.sample(ACTOR_TRAINING_SAMPLE_SIZE,0)
            if rb.index > INITIAL_RANDOM_STEPS and game%TRAIN_ACTOR_EVERY_N_GAME==0 :
                # train_actor_with_advantage(agent,  samples)
                train_actor(agent.actor_model, agent.critic_model, noisy_agent.actor_model, samples, ACTOR_TRAINING_SAMPLE_SIZE, env.action_space.shape[0])
        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        all_scores.append(score)
        if (game%PRINT_EVERY==0):
            plot_score(all_scores)
            print("episide ", game,"score",score ,"episode_len", len(episode_sars),"buffer",min(rb.index,rb.buffer_size), "score", np.average( avg_reward), "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
