from plotting import *
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

from features import feature_engineering
from dqn_model import DQN, gen_epsilon_greedy_policy

from pyvirtualdisplay import Display
disp = Display().start()

def q_learning(env, estimator, n_episode, target_update=5, gamma=1.0, epsilon=0.1, epsilon_decay=0.999):

    for episode in range(n_episode):

        if episode % target_update == 0:
            estimator.copy_target()

        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        #state = feature_engineering(state)
        is_done = False

        saved_rewards = [0, 0, 0, 0]
        running_reward = 0
        pick_ups = 0

        while not is_done:
            action = policy(state)

            if action == 5: 
                saved_rewards[3] += 1

            next_state, reward, is_done, _ = env.step(action)

            if reward == 100: 
                pick_ups += 1
                reward = 10000

            #next_state = feature_engineering(next_state)
            saved_rewards = track_reward(reward, saved_rewards)
            running_reward += reward

            memory.append((state, action, next_state, reward, is_done))

            if is_done:
                
                print(f"Episode: {episode} Reward: {running_reward} Passengers: {pick_ups//2}")
                break
        
            state = next_state

        estimator.replay(memory, replay_size, gamma)
        epsilon = max(epsilon * epsilon_decay, 0.01)

        rewards.append(running_reward)
        illegal_pick_ups.append(saved_rewards[1])
        illegal_moves.append(saved_rewards[2])
        do_nothing.append(saved_rewards[3])
        episolons.append(epsilon)
        n_passengers.append(pick_ups//2)


torch.manual_seed(42)
env_name = "CartPole-v1"
env = gym.make(env_name)

n_action = env.action_space.n
n_episode = 1000
n_feature = env.observation_space.shape[0]
lr = 0.01
n_hidden = 64
memory = deque(maxlen=50000)
replay_size = 10

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


illegal_pick_ups = []
illegal_moves = []
do_nothing = []
episolons = []
n_passengers = []
rewards = []

dqn = DQN(n_feature, n_action, n_hidden, lr)
q_learning(env, dqn, n_episode, target_update=10, gamma=0.99, epsilon=1, epsilon_decay=0.99)

dqn.save_model(log_path)

from plotting import * 

plot_rewards(rewards, log_path)
plot_rewards_and_epsilon(rewards, episolons, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(rewards, illegal_pick_ups, illegal_moves, do_nothing,log_path)
