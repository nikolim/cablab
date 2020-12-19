import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_rewards(rewards):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')
    ax1.plot(rewards)
    plt.savefig('../plots/rewards.png')

def plot_rewards_and_entropy(rewards, entropy):

    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Entropy', color=color)
    ax2.plot(entropy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig('../plots/rewars_entropy.png')

def plot_rewards_and_passengers(rewards, n_passenger):

    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Passengers', color=color)
    ax2.plot(n_passenger, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig('../plots/rewards_passenger.png')


def plot_rewards_and_illegal_actions(rewards, illegal_drop_offs, illegal_moves):

    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Illegal drop offs', color=color)
    ax2.plot(illegal_drop_offs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'darkblue'
    ax2.set_ylabel('Illegal moves', color=color)
    ax2.plot(illegal_moves, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig('../plots/rewards_illegal_actions.png')


def smoothing():
    rewards = savgol_filter(rewards, 5, 3)
    diff = np.subtract(rewards, rewards_copy)
    diff_mean = diff.mean()
    x_axis = range(len(rewards))
    top = [x+diff_mean for x in rewards]
    bottom = [x-diff_mean for x in rewards]
    ax1.fill_between(x_axis, top, bottom, alpha=.5)


plot_rewards([1,2,3,4,5])