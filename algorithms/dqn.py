import os
import gym
import gym_cabworld
import random
import time
import torch
from collections import deque
from tensorboard_tracker import track_reward, log_rewards

from torch.utils.tensorboard import SummaryWriter
from dqn_model import DqnEstimator

from pyvirtualdisplay import Display
disp = Display().start()


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def dqn_learning(env, estimator, n_episode, writer, gamma, epsilon, epsilon_decay, n_action, render):
    total_reward_episode = [0] * n_episode
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** episode, n_action)
        state = env.reset()
        is_done = False
        saved_rewards = (0, 0, 0)
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            saved_rewards = track_reward(reward, saved_rewards)
            q_values_next = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(q_values_next)
            total_reward_episode[episode] += reward
            estimator.update(state, action, td_target, episode)
            if is_done:
                print(
                    f'Episode: {episode} Reward: {total_reward_episode[episode]}')
                log_rewards(writer, saved_rewards,
                            total_reward_episode[episode], episode)
                estimator.total_loss = 0
                estimator.n_updates = 0
                break
            state = next_state


env = gym.make('Cabworld-v4')
n_action = env.action_space.n
n_episode = 3000
n_feature = 11
lr = 0.001

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, '../runs', 'dqn')
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_folders = os.listdir(log_path)
if len(log_folders) == 0:
    folder_number = 0
else:
    folder_number = max([int(elem) for elem in log_folders]) + 1

log_path = os.path.join(log_path, str(folder_number))
writer = SummaryWriter(log_path)

estimator = DqnEstimator(n_feature, n_action, lr, writer)
dqn_learning(env, estimator, n_episode, writer, 0.99, 1, 0.99, n_action, False)
