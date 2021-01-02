
import os
import gym
import torch

from collections import deque
import random

import copy
from torch.autograd import Variable

from tensorboard_tracker import track_reward
from features import feature_engineering

import gym_cabworld

env_name = "CartPole-v1"
env = gym.envs.make(env_name)

class DQN():
    def __init__(self, n_state, n_action, n_hidden=32, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(n_state, n_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, n_action)
                )

        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.losses = []

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def target_predict(self, s):
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)

            states = []
            td_targets = []

            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                td_targets.append(q_values)

            self.update(states, td_targets)

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        full_path = os.path.join(path, 'dqn.pth')
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded {path}")


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function


def q_learning(env, estimator, n_episode, replay_size, target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    for episode in range(n_episode):

        global counter 
        counter += 1

        if episode % target_update == 0:
            estimator.copy_target()

        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = tuple((list(env.reset()))[:n_state])
        is_done = False

        saved_rewards = [0, 0, 0, 0]
        running_reward = 0
        pick_ups = 0
        number_of_action_4 = 0
        number_of_action_5 = 0

        while not is_done:

            action = policy(state)

            if action == 4: 
                number_of_action_4 += 1
            
            if action == 5: 
                number_of_action_5 += 1

            if action == 6: 
                saved_rewards[3] += 1

            next_state, reward, is_done, _ = env.step(action)
            next_state = tuple((list(next_state))[:n_state])
            saved_rewards = track_reward(reward, saved_rewards)

            if reward == 100: 
                pick_ups += 1
                reward = 1000

            running_reward += reward
            
            memory.append((state, action, next_state, reward, is_done))

            estimator.replay(memory, replay_size, gamma)

            if is_done:
                print(f"Episode: {episode} Reward: {running_reward} Passengers: {pick_ups//2} N-Action-4: {number_of_action_4} N-Action-5: {number_of_action_5}") 
                break

            state = next_state

        epsilon = max(epsilon * epsilon_decay, 0.01)

        rewards.append(running_reward)
        illegal_pick_ups.append(saved_rewards[1])
        illegal_moves.append(saved_rewards[2])
        do_nothing.append(saved_rewards[3])
        episolons.append(epsilon)
        n_passengers.append(pick_ups//2)


counter = 0

n_state = 4
n_action = 2
n_hidden = 32
lr = 0.01

n_episode = 10
replay_size = 10
target_update = 5

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
episolons = []
n_passengers = []
rewards = []

dqn = DQN(n_state, n_action, n_hidden, lr)
memory = deque(maxlen=50000)

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "dqn")
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_folders = os.listdir(log_path)
if len(log_folders) == 0:
    folder_number = 0
else:
    folder_number = max([int(elem) for elem in log_folders]) + 1

log_path = os.path.join(log_path, str(folder_number))
os.mkdir(log_path)
with open(os.path.join(log_path, "info.txt"), "w+") as info_file: 
    info_file.write(env_name + "\n")
    info_file.write("Episodes:" + str(n_episode) + "\n")


q_learning(env, dqn, n_episode, replay_size, target_update, gamma=.99, epsilon=1)
dqn.save_model(log_path)

from plotting import * 

plot_rewards(rewards, log_path)
plot_rewards_and_epsilon(rewards, episolons, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(rewards, illegal_pick_ups, illegal_moves, do_nothing,log_path)
