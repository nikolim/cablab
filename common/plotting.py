import os
from scipy.signal.filter_design import EPSILON
from scipy.signal.ltisys import step
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.colors as mc
import colorsys

REWARD_COLOR = "royalblue"
PASSENGER_COLOR = "forestgreen"
ENTROPY_COLOR = "darkorange"
EPSILON_COLOR = "orange"
ILLEGAL_MOVE_COLOR = "peru"
ILLEGAL_PICK_UP_COLOR = "sandybrown"
DO_NOTHING_COLOR = "salmon"
STEPS_BETWEEN_PASSENGERS = "peru"
STANDARD_COLOR = "blue"
OPT_PICK_UP_COLOR = "olive"

# Seaborn backend
sns.set()


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def smoothing_mean_std(arr, step_size):

    arr = np.array(arr)
    mean_arr = np.array([])
    std_arr = np.array([])

    for i in range(0, len(arr), step_size): 
        array_slice = arr[i:i+step_size]
        mean_arr = np.append(mean_arr, array_slice.mean())
        std_arr = np.append(std_arr, array_slice.std())

    x_values = np.array(list(range(step_size, len(arr)+step_size, step_size)))

    return mean_arr, std_arr, x_values


def plot_rewards(rewards, path):

    mean_rewards, std_rewards, x = smoothing_mean_std(rewards, step_size=10)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward")
    ax1.plot(x, mean_rewards, color=REWARD_COLOR)
    ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=REWARD_COLOR)
    plt.savefig(os.path.join(path, "rewards.png"))


def plot_opt_pick_ups(opt_pick_ups, path):

    mean_opt_pick_ups, std_opt_pick_ups, x = smoothing_mean_std(opt_pick_ups, step_size=10)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Optimal Pick-up")
    ax1.plot(x, mean_opt_pick_ups, color=OPT_PICK_UP_COLOR)
    ax1.fill_between(x, mean_opt_pick_ups + std_opt_pick_ups, mean_opt_pick_ups - std_opt_pick_ups, alpha=0.2, color=OPT_PICK_UP_COLOR)
    plt.savefig(os.path.join(path, "opt_pick_ups.png"))


def plot_rewards_and_entropy(rewards, entropy, path):

    mean_rewards, std_rewards, x = smoothing_mean_std(rewards, step_size=10)
    mean_entropy, std_entropy, x = smoothing_mean_std(entropy, step_size=10)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis="y")
    ax2.tick_params(axis="y")

    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward", color=REWARD_COLOR)
    ax1.plot(x, mean_rewards, color=REWARD_COLOR)
    ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=REWARD_COLOR)
    ax2.set_ylabel("Entropy", color=ENTROPY_COLOR)
    ax2.plot(x, mean_entropy, color=ENTROPY_COLOR)
    ax1.fill_between(x, mean_entropy + std_entropy, mean_entropy - std_entropy, alpha=0.2, color=REWARD_COLOR)

    fig.tight_layout()
    plt.savefig(os.path.join(path, "rewards_entropy.png"))


def plot_rewards_and_epsilon(rewards, epsilon, path):

    mean_rewards, std_rewards, x = smoothing_mean_std(rewards, step_size=10)
    mean_epsilon, std_epsilon, x = smoothing_mean_std(epsilon, step_size=10)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis="y")
    ax2.tick_params(axis="y")

    ax1.set_xlabel("Episodes")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward", color=REWARD_COLOR)
    ax1.plot(x, mean_rewards, color=REWARD_COLOR)
    ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=REWARD_COLOR)

    ax2.set_ylabel("Epsilon", color=EPSILON_COLOR)
    ax2.plot(x, mean_epsilon, color=EPSILON_COLOR)
    ax2.fill_between(x, mean_epsilon + std_epsilon, mean_epsilon - std_epsilon, alpha=0.2, color=EPSILON_COLOR)

    fig.tight_layout()
    plt.savefig(os.path.join(path, "rewards_epsilon.png"))


def plot_rewards_and_passengers(rewards, n_passenger, path):

    mean_rewards, std_rewards, x = smoothing_mean_std(rewards, step_size=10)
    mean_pass, std_pass, x = smoothing_mean_std(n_passenger, step_size=10)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis="y")
    ax2.tick_params(axis="y")

    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward", color=REWARD_COLOR)
    ax1.plot(x, mean_rewards, color=REWARD_COLOR)
    ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=REWARD_COLOR)

    ax2.set_ylabel("Passengers", color=PASSENGER_COLOR)
    ax2.plot(x, mean_pass, color=PASSENGER_COLOR)
    ax2.fill_between(x, mean_pass + std_pass, mean_pass - std_pass, alpha=0.2, color=PASSENGER_COLOR)

    fig.tight_layout()
    plt.savefig(os.path.join(path, "rewards_passengers.png"))


def plot_do_nothing(do_nothing, do_nothing_opt, do_nothing_sub, path):

    mean_do, std_do, x = smoothing_mean_std(do_nothing, step_size=10)
    mean_opt, std_opt, x = smoothing_mean_std(do_nothing_opt, step_size=10)
    mean_sub, std_sub, x = smoothing_mean_std(do_nothing_sub, step_size=10)

    fig, ax1 = plt.subplots()
    ax1.tick_params(axis="y")

    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Do nothing")

    ax1.plot(x, mean_do, color=ENTROPY_COLOR, label="Do nothing")
    ax1.fill_between(x, mean_do + std_do, mean_do - std_do, alpha=0.2, color=ENTROPY_COLOR)

    ax1.plot(x, mean_opt, color=EPSILON_COLOR, label="Optimal")
    ax1.fill_between(x, mean_opt + std_opt, mean_opt - std_opt, alpha=0.2, color=EPSILON_COLOR)

    ax1.plot(x, mean_sub, color=ILLEGAL_MOVE_COLOR, label="Suboptimal")
    ax1.fill_between(x, mean_sub + std_sub, mean_sub - std_sub, alpha=0.2, color=ILLEGAL_MOVE_COLOR)

    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "do_nothing_steps.png"))



def plot_multiple_agents(data, description, path, sum=True):

    if description == "rewards": 
        color = REWARD_COLOR
    elif description == "passengers": 
        color = PASSENGER_COLOR 
    else: 
        color = STANDARD_COLOR

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel(description)

    if sum:
        np_arr = [np.array([tmp]) for tmp in data]
        summed = np.sum(np_arr, axis=0)

        total_color = adjust_lightness(color, 0.5)

        mean_rewards, std_rewards, x = smoothing_mean_std(summed.T, step_size=10)
        ax1.plot(x, mean_rewards, color=total_color, label="Total")
        ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=total_color)

    for i, rewards in enumerate(data):
        mean_rewards, std_rewards, x = smoothing_mean_std(rewards, step_size=10)
        ax1.plot(x, mean_rewards, color=color, label=("Agent " + str(i)))
        ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=color)
    
    plt.legend(loc="best")
    plt.savefig(os.path.join(path, "multiple_" + str(description) + ".png"))


def plot_rewards_and_illegal_actions(rewards, illegal_drop_offs, illegal_moves, path):

    mean_rewards, std_rewards, x = smoothing_mean_std(rewards, step_size=10)
    mean_drop, std_drop, x = smoothing_mean_std(illegal_drop_offs, step_size=10)
    mean_moves, std_moves, x = smoothing_mean_std(illegal_moves, step_size=10)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis="y")
    ax2.tick_params(axis="y")

    ax1.set_xlabel("Episodes")
    ax1.plot(x, mean_rewards, color=REWARD_COLOR)
    ax1.fill_between(x, mean_rewards + std_rewards, mean_rewards - std_rewards, alpha=0.2, color=REWARD_COLOR)

    ax2.set_ylabel("Moves")

    ax2.plot(x, mean_drop, color=ILLEGAL_PICK_UP_COLOR, label="Illegal Drop-Offs")
    ax2.fill_between(x, mean_drop + std_drop, mean_drop - std_drop, alpha=0.2, color=ILLEGAL_PICK_UP_COLOR)
    ax2.plot(x, mean_moves, color=ILLEGAL_MOVE_COLOR, label="Illegal Moves")
    ax2.fill_between(x, mean_moves + std_moves, mean_moves - std_moves, alpha=0.2, color=ILLEGAL_MOVE_COLOR)

    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "rewards_illegal_actions.png"))


def plot_mean_pick_up_drop_offs(pick_up_mean, drop_off_mean, path):

    mean_pick, std_pick, x = smoothing_mean_std(pick_up_mean, step_size=10)
    mean_drop, std_drop, x = smoothing_mean_std(drop_off_mean, step_size=10)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.tick_params(axis="y")
    ax2.tick_params(axis="y")

    ax1.set_xlabel("Episodes")

    ax1.set_ylabel("Pick-Up-Steps", color=ILLEGAL_PICK_UP_COLOR)
    ax1.plot(x, mean_pick, color=ILLEGAL_PICK_UP_COLOR, label="Pick-Up-Steps")
    ax1.fill_between(x, mean_pick + std_pick, mean_pick - std_pick, alpha=0.2, color=ILLEGAL_PICK_UP_COLOR)

    ax2.set_ylabel("Drop-Off-Steps", color=ILLEGAL_MOVE_COLOR)
    ax2.plot(x, mean_drop, color=ILLEGAL_MOVE_COLOR, label="Drop-Off-Steps")
    ax2.fill_between(x, mean_drop + std_drop, mean_drop - std_drop, alpha=0.2, color=ILLEGAL_MOVE_COLOR)


    # plt.legend(loc='best')
    fig.tight_layout()
    plt.savefig(os.path.join(path, "pick_up_drop_off_path.png"))


