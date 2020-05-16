import gym
from DQN_torch import Agent
# from utils import plotLearning
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ =='__main__':
    env=gym.make('LunarLander-v2')
    brain = Agent(gamma=0.98, epsilon=0.7, batch_size=2, n_actions=4,
                input_dims=[8], lr=0.01, eps_end=0.02,eps_dec=0.992)
    scores = []
    eps_history = []
    n_games = 50000
    score = 0

    for i in range (n_games):
        if i%10 == 0 and i>0 and i>10:
            avg_score = np.mean(scores[:-10])
            print('epside ', i, 'score', score,
            'average score %3f' % avg_score,
            'epsilon %3f' % brain.epsilon)
        else:
            print('episode ',i, 'score', score)
        score = 0

        eps_history.append(brain.epsilon)
        observation = env.reset()
        done = False
        step_counter = 0
        while not done:
            env.render()
            action = brain.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # print("chosen action after chose action ", action)
            score+=reward
            brain.store_transition(observation,action, reward, observation_,done)
            if i>5:
                brain.learn(step_counter)
            observation = observation_
            step_counter += 1
        # EPISODE done
        # "CALCULATE BELL MAN IN AGENT CLASS brain.compute_reward()"
        brain.process_end_of_episode(step_counter)
        scores.append(score)

    # x = [i+1 for i in range(n_games)]
    # filename = 'lunar-lander.png'
    # plotLearning(x, scores, eps_history, filename,)
