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


def get_one_hot(action,n_dim):
    retval = np.zeros(n_dim)
    retval[action] = 1.0
    return retval




if __name__=='__main__':
    DEBUGER_ON = True
    NUM_GAMES = 50000
    MAX_EPISODE_STEPS = 10000
    TARGET_MODEL_UPDATE_INTERVAL = 50
    EPSILON_MIN = 0.05
    EPSILON_START = 0.3
    EPSLILON_COUNT = 4000 #Games
    RANDOM_GAME_EVERY = 10
    TRAIN_EVERY_N_STEPS = 10
    TRAINING_SAMPLE_SIZE = 1
    TRAINING_ITTERATIONS = 1
    PRINT_EVERY = 2
    RENDER_ENV = True
    LOAD_MODEL = True
    SAVE_MODEL = True
    MODEL_FILE_NAME = "TDQN_RL_MODEL.trl"
    MODEL_ID = "01"
    SAVE_MODEL_EVERY = 25

    epsilon = EPSILON_START
    env = gym.make('Pong-v0')
    # env = gym.make('CartPole-v1')

    agent = DQNAgent(Model(env.observation_space.shape,env.action_space.n,lr=0.0001), Model(env.observation_space.shape,env.action_space.n,lr=0.0001) )

    observation = env.reset()
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
        print("Loading Model ", ""+MODEL_ID+MODEL_FILE_NAME)
        agent.model = torch.load(""+MODEL_ID+MODEL_FILE_NAME)
        # agent.model.load_state_dict(torch.load(""+MODEL_ID+MODEL_FILE_NAME))
        agent.model.eval()
    step_counter = 0
    avg_reward = []



    for game in range (NUM_GAMES):
        episode_steps = 0
        score = 0.0
        for step in range (MAX_EPISODE_STEPS):
            if RENDER_ENV:
                env.render()
            # import ipdb; ipdb.set_trace()
            action = 0

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

            reward *= 100
            avg_reward.append([reward])

            observation = observation_next
            step_counter+=1
            episode_steps = step
            if done:
                observation = env.reset()
                frame1 = []
                frame2 = []
                frame3 = []
                frame1 = agent.process_frame(observation)
                frame2 = agent.process_frame(observation)
                frame3 = agent.process_frame(observation)
                # import ipdb; ipdb.set_trace()
                observation = np.concatenate((frame1,frame2,frame3),axis=1)
                observation = observation.reshape((1,3,160,140*3))
                break


        epsilon = max(EPSILON_MIN, epsilon-((EPSILON_START-EPSILON_MIN)/EPSLILON_COUNT) )
        if (game%PRINT_EVERY==0):
            print("episide ", game,"last score",reward ,"episode_len", episode_steps , "score", np.average( avg_reward), "epsilon",epsilon )
        avg_reward = []
        # print("epsilon ", epsilon)
