
# DNQ Implementation based on PyTorch example

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gym_cabworld
env = gym.make('CartPole-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_state, n_actions, n_hidden=32):
        super(DQN, self).__init__()

        self.input = torch.nn.Linear(n_state, n_hidden)
        self.hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.input(x.float()))
        x = F.relu(self.hidden(x))
        x = F.relu(self.output(x))
        x = x.view(x.size(0), -1)
        return x


BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.99
TARGET_UPDATE = 5

n_actions = 2
n_state = 4 

policy_net = DQN(n_state, n_actions).to(device)
target_net = DQN(n_state, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(500000)

eps_threshold = EPS_START

def select_action(state, episode):
    global eps_threshold
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START * (EPS_DECAY**episode))
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.argmax()
            return action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    next_state_batch = torch.stack(batch.next_state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 10000

for i_episode in range(num_episodes):
    
    state = env.reset()
    state = (torch.tensor(state))[:n_state]

    running_reward = 0

    for t in count():
        # Select and perform an action
        action = select_action(state, i_episode)
        next_state, reward, done, _ = env.step(action.item())

        running_reward += reward

        action = (torch.tensor([action.item()], device=device))
        next_state = (torch.tensor(next_state, device=device))[:8]
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, next_state, reward)
        state = next_state
        
        optimize_model()

        if done:
            print(f'Episode: {i_episode} Reward: {running_reward} Episilon: {eps_threshold}')
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

