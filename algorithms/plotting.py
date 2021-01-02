import os
from scipy.signal.filter_design import EPSILON
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


REWARD_COLOR = 'royalblue'
PASSENGER_COLOR = 'forestgreen'
ENTROPY_COLOR = 'darkorange'
EPSILON_COLOR = 'orange'
ILLEGAL_MOVE_COLOR = 'peru'
ILLEGAL_PICK_UP_COLOR = 'sandybrown'
DO_NOTHING_COLOR = 'salmon'

# Seaborn backend
sns.set()

def plot_rewards(rewards, path):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')
    ax1.plot(rewards, color=REWARD_COLOR)
    plt.savefig(os.path.join(path,'rewards.png'))

def plot_rewards_and_entropy(rewards, entropy, path):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=REWARD_COLOR)
    ax1.plot(rewards, color=REWARD_COLOR)
    ax2.set_ylabel('Entropy', color=ENTROPY_COLOR)
    ax2.plot(entropy, color=ENTROPY_COLOR)
    

    fig.tight_layout()
    plt.savefig(os.path.join(path,'rewards_entropy.png'))


def plot_rewards_and_epsilon(rewards, epsilon, path):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=REWARD_COLOR)
    ax1.plot(rewards, color=REWARD_COLOR)
    ax2.set_ylabel('Epsilon', color=EPSILON_COLOR)
    ax2.plot(epsilon, color=EPSILON_COLOR)
    

    fig.tight_layout()
    plt.savefig(os.path.join(path,'rewards_epsilon.png'))

def plot_rewards_and_passengers(rewards, n_passenger, path):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=REWARD_COLOR)
    ax1.plot(rewards, color=REWARD_COLOR)
    ax2.set_ylabel('Passengers', color=PASSENGER_COLOR)
    ax2.plot(n_passenger, color=PASSENGER_COLOR)
    
    fig.tight_layout()
    plt.savefig(os.path.join(path,'rewards_passengers.png'))


def plot_rewards_and_illegal_actions(rewards, illegal_drop_offs, illegal_moves, do_nothing, path):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=REWARD_COLOR)
    ax1.plot(rewards, color=REWARD_COLOR, label='Rewards')

    ax2.set_ylabel('Moves')
    ax2.plot(illegal_drop_offs, color=ILLEGAL_PICK_UP_COLOR, label='Illegal Drop-Offs')
    ax2.plot(illegal_moves, color=ILLEGAL_MOVE_COLOR, label='Illegal Moves')
    ax2.plot(do_nothing, color=DO_NOTHING_COLOR,label='Do nothing')
    
    plt.legend(loc='best')
    fig.tight_layout()
    plt.savefig(os.path.join(path,'rewards_illegal_actions.png'))


test = np.array([1,2,3,4,5,6,7,8,9])
test2 = np.invert(test)

plot_rewards_and_entropy(test, test2, '.')