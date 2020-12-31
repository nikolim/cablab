import os
import gym
import gym_cabworld
import random
import time
import torch
from collections import deque
from tensorboard_tracker import track_reward, log_rewards, log_reward_uncertainty

from torch.utils.tensorboard import SummaryWriter
from a2c_model import ActorCriticModel, PolicyNetwork

from pyvirtualdisplay import Display

disp = Display().start()


def actor_critic(
    env, estimator, n_episode, writer, gamma, epsilon, epsilon_decay, n_action, render
):

    for episode in range(n_episode):
        log_probs = []
        rewards = []
        running_reward = 0
        state_values = []
        state = env.reset()
        state = tuple((list(state))[:8])
        saved_rewards = (0, 0, 0)
        pick_ups = 0

        while True:
            action, log_prob, state_value = estimator.get_action(state)
            next_state, reward, is_done, _ = env.step(action)
            next_state = tuple((list(next_state))[:8])

            reward += 1

            running_reward += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)

            if reward == 100: 
                pick_ups += 1

            if is_done:
                print(f"Episode: {episode} Reward: {running_reward} Passengers {pick_ups//2}")
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs, state_values, episode)
                uncertainty = sum(log_probs) * -1
                log_reward_uncertainty(writer, running_reward, uncertainty, episode)
                break
            state = next_state


env = gym.make("Cabworld-v6")
n_action = 4
n_episode = 100
n_feature = 8
lr = 0.01

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "a2c")
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_folders = os.listdir(log_path)
if len(log_folders) == 0:
    folder_number = 0
else:
    folder_number = max([int(elem) for elem in log_folders]) + 1

log_path = os.path.join(log_path, str(folder_number))
writer = SummaryWriter(log_path)

estimator = PolicyNetwork(writer)
actor_critic(env, estimator, n_episode, writer, 0.99, 1, 0.99, n_action, False)
