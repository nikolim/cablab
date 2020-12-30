import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_rewards(rewards, path):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')
    ax1.plot(rewards)
    plt.savefig(os.path.join(path,'rewards.png'))

def plot_rewards_and_entropy(rewards, entropy, path):

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
    plt.savefig(os.path.join(path,'rewards_entropy.png'))


def plot_rewards_and_epsilon(rewards, epsilon, path):

    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(epsilon, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(os.path.join(path,'rewards_epsilon.png'))

def plot_rewards_and_passengers(rewards, n_passenger, path):

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
    plt.savefig(os.path.join(path,'rewards_passengers.png'))


def plot_rewards_and_illegal_actions(rewards, illegal_drop_offs, illegal_moves, do_nothing, path):

    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color, label='Rewards')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Moves', color='black')
    line1 = ax2.plot(illegal_drop_offs, color=color, label='Illegal drop-offs')
    ax2.tick_params(axis='y', labelcolor='black')

    color = 'darkblue'
    line2 = ax2.plot(illegal_moves, color=color, label='Illegal moves')
    ax2.tick_params(axis='y', labelcolor='black')

    color = 'green'
    line3 = ax2.plot(do_nothing, color=color,label='Do nothing')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(path,'rewards_illegal_actions.png'))

def smoothing():
    rewards = savgol_filter(rewards, 5, 3)
    diff = np.subtract(rewards, rewards_copy)
    diff_mean = diff.mean()
    x_axis = range(len(rewards))
    top = [x+diff_mean for x in rewards]
    bottom = [x-diff_mean for x in rewards]
    ax1.fill_between(x_axis, top, bottom, alpha=.5)