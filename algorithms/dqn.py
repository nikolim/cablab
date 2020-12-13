import os
import gym
import gym_cabworld
import random
import time
import torch
from torch.autograd import Variable
from collections import deque
from tensorboard_tracker import track_reward, log_rewards, log_reward_epsilon
from torch.utils.tensorboard import SummaryWriter

from dqn_model import DQN, gen_epsilon_greedy_policy

from pyvirtualdisplay import Display

disp = Display().start()


def q_learning(
    env, estimator, n_episode, writer, gamma=1.0, epsilon=0.1, epsilon_decay=0.99
):
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        is_done = False
        saved_rewards = (0, 0, 0)
        running_reward = 0
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            saved_rewards = track_reward(reward, saved_rewards)
            running_reward += reward
            q_values = estimator.predict(state).tolist()

            if is_done:
                q_values[action] = reward
                estimator.update(state, q_values)
                print(f"Episode: {episode} Reward: {running_reward}")
                log_rewards(writer, saved_rewards, running_reward, episode)
                log_reward_epsilon(writer, running_reward, epsilon, episode)
                break

            q_values_next = estimator.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            estimator.update(state, q_values)
            state = next_state

        epsilon = max(epsilon * epsilon_decay, 0.01)


env = gym.make("Cabworld-v5")
n_action = env.action_space.n
n_episode = 3000
n_feature = 11
lr = 0.001
n_hidden = 64

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
writer = SummaryWriter(log_path)

dqn = DQN(n_feature, n_action, n_hidden, lr)
q_learning(env, dqn, n_episode, writer, gamma=0.9, epsilon=1)
