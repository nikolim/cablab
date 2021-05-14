import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.colors as mc
import colorsys
import pandas as pd

color_dict = {
    "rewards": "royalblue",
    "n_passengers": "forestgreen",
    "illegal_pick_ups": "sandybrown",
    "illegal_moves": "peru",
    "avg_waiting_time": "darkorchid",
    "mean_actions_per_pick_up": "turquoise"
}

# Seaborn backend
sns.set()
plt.figure(dpi=1200)

smoothing_factor = 50


def adjust_lightness(color, amount=0.5):
    """
    Make color darker or ligther
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def smoothing_mean_std(arr, step_size):
    """
    Smoothing over array and calculate standard deviation
    """
    arr = np.array(arr)
    mean_arr = np.array([])
    std_arr = np.array([])

    for i in range(0, len(arr), step_size):
        array_slice = arr[i: i + step_size]
        mean_arr = np.append(mean_arr, array_slice.mean())
        std_arr = np.append(std_arr, array_slice.std())

    x_values = np.array(
        list(range(step_size, len(arr) + step_size, step_size)))

    return mean_arr, std_arr, x_values

def smoothing(arr, step_size):
    """
    Smoothing over array
    """
    arr = np.array(arr)

    for i in range(0, len(arr), step_size):
        array_slice = arr[i: i + step_size]
        mean_arr = np.append(mean_arr, array_slice.mean())

    x_values = np.array(
        list(range(step_size, len(arr) + step_size, step_size)))

    return mean_arr, x_values


def plot_losses(arr):
    """
    Plot training losses 
    """
    fig, ax1 = plt.subplots()
    mean_rewards, std_rewards, x = smoothing_mean_std(
        arr, step_size=smoothing_factor)
    ax1.plot(mean_rewards)
    plt.savefig(os.path.join("losses.png"), dpi=1200)


def plot_values(df, ids, path):
    """
    Plot values of pandas datagrame for single agent
    """
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")

    for id in ids:
        data = df[id]
        mean_rewards, std_rewards, x = smoothing_mean_std(
            data, step_size=smoothing_factor)

        try: 
            color = color_dict[id]
        except KeyError: 
            color = 'black'

        ax1.plot(x, mean_rewards, color=color, label=id)
        ax1.fill_between(
            x,
            mean_rewards + std_rewards,
            mean_rewards - std_rewards,
            alpha=0.2,
            color=color,
        )
    if len(ids) == 1:
        ax1.set_ylabel(ids[0])
    else:
        plt.legend(loc="best")

    fig.tight_layout()
    name = "_".join(ids)
    plt.savefig(os.path.join(path, name + ".png"), dpi=1200)


def plot_mult_agent(dfs, ids, path, labels=None, colors=None):
    """
    Plot values of pandas dataframe for mulit agent
    """
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")

    if colors:
        assert len(colors) == len(dfs)

    for id in ids:

        try:
            color = color_dict[id]
        except KeyError: 
            color = 'black'
        amount = 1

        for i, df in enumerate(dfs):
            data = df[id]
            color = adjust_lightness(color_dict[id], amount)
            if colors:
                color = colors[i]
            mean_metric, std_metric, x = smoothing_mean_std(
                data, step_size=smoothing_factor)
            ax1.set_ylabel(id)
            label = labels[i] if labels else (id + str(i))
            ax1.plot(x, mean_metric, color=color, label=label)
            ax1.fill_between(
                x,
                mean_metric + std_metric,
                mean_metric - std_metric,
                alpha=0.2,
                color=color,
            )
            amount -= 0.5 / len(dfs)

    plt.legend(loc="best")
    fig.tight_layout()
    name = "_".join(ids)

    plt.savefig(os.path.join(path, name + "_mult.png"), dpi=1200)


def plot_mult_runs(dfs, ids, path, labels=None, double_scale=False):
    """
    Plot values of multiple runs
    """
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")

    DF = pd.concat(dfs, keys=range(len(dfs))).groupby(level=1)

    meandf = DF.mean()
    maxdf = DF.max()
    mindf = DF.min()
    
    for i, id in enumerate(ids):
        try:
            color = color_dict[id]
        except KeyError: 
            color = 'black'
        ax1.set_ylabel(id)

        mean_data, x = smoothing(meandf[id], step_size=smoothing_factor)
        min_data, x = smoothing(mindf[id], step_size=smoothing_factor)
        max_data, x = smoothing(maxdf[id], step_size=smoothing_factor)
            

        ax1.plot(x, mean_data, color=color, label="Mean")
        ax1.fill_between(
            x,
            min_data,
            max_data,
            alpha=0.2,
            color=color,
        )

    plt.legend(loc="best")
    fig.tight_layout()
    name = "_".join(ids)

    file_name = os.path.join(path, "multi_run_df.csv")
    DF.to_csv(file_name)

    plt.savefig(os.path.join(path, name + "_mult_runs.png"), dpi=1200)


def plot_arr(arr, path, name):
    fig, ax1 = plt.subplots()
    ax1.plot(arr[1:])
    plt.savefig(os.path.join(path, name), dpi=1200)
