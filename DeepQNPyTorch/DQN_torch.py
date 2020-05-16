import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,n_actions):
        super(DeepQNetwork, self).__init__()
        # print("input_dims ", input_dims[0], " n_actions ",n_actions)
        self.input_dims = input_dims[0]+n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = 1
        self.fc1 = nn.Linear(self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        if torch.cuda.is_available():
            print("Using CUDA")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

        # print("Cuda device ",self.device)
        self.to(self.device)

    def forward(self,state_action):
        #action to 1 hot
        # action_1hot = np.zeros(self.n_actions)
        # action_1hot[action] = 1.0
        # observation_state = np.append(observation,action_1hot)
        state = torch.Tensor(state_action).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu((self.fc2(x)))
        predicted_Q_value = self.fc3(x)
        return predicted_Q_value


class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                max_mem_size=100000, eps_end=0.01, eps_dec=0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        # print("input_dims ", input_dims)
        self.input_dims = input_dims[0]
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.max_mem_size = max_mem_size
        self.mem_counter = 0
        self.Q_eval = DeepQNetwork(lr=lr,n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.max_mem_size,self.input_dims))
        self.new_state_memory = np.zeros((self.max_mem_size,self.input_dims))
        self.action_memory = np.zeros((self.max_mem_size,n_actions))
        self.action_state_memory = np.zeros((self.max_mem_size,self.input_dims+n_actions))
        self.reward_memory = np.zeros(self.max_mem_size)
        self.Q_memory = np.zeros(self.max_mem_size)
        self.terminal_memory = np.zeros(self.max_mem_size)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.max_mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
        self.new_state_memory[index] = state_
        self.action_state_memory[index] = np.append(state,actions)
        self.mem_counter+=1

    def calculate_bellman(self,episode_len):
        for i in range(episode_len):
            index = ((self.mem_counter-1)-i) % self.max_mem_size
            next_index = ((self.mem_counter)-i) % self.max_mem_size
            if i==0:
                self.Q_memory[index] = self.reward_memory[index]
                # print("last Q ", self.Q_memory[index])
            else:
                self.Q_memory[index] = self.reward_memory[index] + self.gamma * self.Q_memory[next_index]
                # print("Q ", self.Q_memory[index])

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def process_end_of_episode(self,episode_len):
        self.calculate_bellman(episode_len)
        self.update_epsilon()

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            estimated_Q_values = torch.Tensor( np.zeros(self.n_actions)).to(self.Q_eval.device)
            for i in range (self.n_actions):
                action_1hot = np.zeros(self.n_actions)
                action_1hot[i] = 1.0
                # print("about to evaulate action ",i, " array of action ", action_1hot," with observation",  observation[:10])
                # print("concatanated sate action vector ", np.append(observation,action_1hot) )
                estimated_Q_values[i] = self.Q_eval.forward(  np.append(observation,action_1hot)  )


            # actions = self.Q_eval.forward(observation)

            # print("estimated Q values", estimated_Q_values)
            action = torch.argmax(estimated_Q_values).item()
        return action

    def learn(self, step_counter):
        if self.mem_counter > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = self.mem_counter if self.mem_counter < self.max_mem_size else self.max_mem_size
            batch = np.random.choice(max_mem,self.batch_size)
            #print("batch size", batch.size())
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.uint8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            q_batch = self.Q_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            action_state_batch = self.action_state_memory[batch]

            reward_batch = torch.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = torch.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(action_state_batch).to(self.Q_eval.device)
            q_target = torch.Tensor(q_batch).to(self.Q_eval.device)
            # q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            # batch_index = np.arange(self.batch_size, dtype=np.int32)
            # q_target[action_batch] = reward_batch  + self.gamma*torch.max(q_next, dim=1)[0]*terminal_batch

            # self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_end else self.eps_end
            # if (step_counter%50)==49:
            #     print("Q eval ",q_eval, "q target", q_target)
            loss = self.Q_eval.loss(q_eval,q_target)
            loss.backward()
            self.Q_eval.optimizer.step()
print(torch.cuda.is_available())
