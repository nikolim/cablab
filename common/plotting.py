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
    "do_nothing_arr": "salmon",
    "do_nothing_opt_arr": "orchid",
    "do_nothing_sub_arr": "blueviolet",
    "epsilon": "orange",
    "useless_steps": "turquoise",
    "wrong_psng": "red", 
    "assigned_psng": "green", 
    "avg_waiting_time": "green", 
}

# Seaborn backend
sns.set()

plt.figure(dpi=1200)


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
        array_slice = arr[i : i + step_size]
        mean_arr = np.append(mean_arr, array_slice.mean())
        std_arr = np.append(std_arr, array_slice.std())

    x_values = np.array(list(range(step_size, len(arr) + step_size, step_size)))

    return mean_arr, std_arr, x_values

def plot_losses(arr): 
    fig, ax1 = plt.subplots()
    mean_rewards, std_rewards, x = smoothing_mean_std(arr, step_size=50)
    ax1.plot(mean_rewards)
    plt.savefig(os.path.join("losses.png"), dpi=1200)

def plot_values(df, ids, path, double_scale=False):

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")

    if double_scale:
        assert len(ids) == 2
        ax2 = ax1.twinx()
        id = ids[0]
        data = df[id]
        mean_rewards, std_rewards, x = smoothing_mean_std(data, step_size=50)
        ax1.set_ylabel(id, color=color_dict[id])
        ax1.plot(x, mean_rewards, color=color_dict[id], label=id)
        ax1.fill_between(
            x,
            mean_rewards + std_rewards,
            mean_rewards - std_rewards,
            alpha=0.2,
            color=color_dict[id],
        )
        id = ids[1]
        data = df[id]
        mean_rewards, std_rewards, x = smoothing_mean_std(data, step_size=50)
        ax2.set_ylabel(id, color=color_dict[id])
        ax2.plot(x, mean_rewards, color=color_dict[id], label=id)
        ax2.fill_between(
            x,
            mean_rewards + std_rewards,
            mean_rewards - std_rewards,
            alpha=0.2,
            color=color_dict[id],
        )
    else:
        for id in ids:
            data = df[id]
            mean_rewards, std_rewards, x = smoothing_mean_std(data, step_size=50)

            ax1.plot(x, mean_rewards, color=color_dict[id], label=id)
            ax1.fill_between(
                x,
                mean_rewards + std_rewards,
                mean_rewards - std_rewards,
                alpha=0.2,
                color=color_dict[id],
            )

    if len(ids) == 1:
        ax1.set_ylabel(ids[0])
    else:
        plt.legend(loc="best")

    fig.tight_layout()
    name = "_".join(ids)
    plt.savefig(os.path.join(path, name + ".png"), dpi=1200)


def plot_mult_agent(dfs, ids, path, labels=None, double_scale=False):

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")

    for id in ids:

        color = color_dict[id]
        amount = 1

        if double_scale:
            assert len(ids) == 2
            ax2 = ax1.twinx()

            for i, df in enumerate(dfs):
                id = ids[0]
                data = df[id]
                #color = adjust_lightness(color_dict[id], amount)
                if i != 0: 
                    color = "blueviolet"

                mean_metric, std_metric, x = smoothing_mean_std(data, step_size=50)
                ax1.set_ylabel(id)
                label = labels[i] if labels else (id + str(i))
                ax1.plot(x, mean_metric, color=color, label=label)
                ax1.fill_between(
                    x,
                    mean_metric + std_metric,
                    mean_metric - std_metric,
                    alpha=0.2,
                    color=color_dict[id],
                )

                id = ids[1]
                data = df[id]
                color = adjust_lightness(color_dict[id], amount)

                mean_metric, std_metric, x = smoothing_mean_std(data, step_size=50)
                ax2.set_ylabel(id)
                label = labels[i] if labels else (id + str(i))
                ax2.plot(x, mean_metric, color=color, label=label)
                ax2.fill_between(
                    x,
                    mean_metric + std_metric,
                    mean_metric - std_metric,
                    alpha=0.2,
                    color=color,
                )
                
                amount -= 0.5 / len(dfs)
        else: 
            for i, df in enumerate(dfs):
                data = df[id]
                color = adjust_lightness(color_dict[id], amount)
                mean_metric, std_metric, x = smoothing_mean_std(data, step_size=50)
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

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")
    
    DF = pd.concat(dfs,keys=range(len(dfs))).groupby(level=1)

    meandf = DF.mean()
    maxdf= DF.max()
    mindf= DF.min()

    for i, id in enumerate(ids):
        color = color_dict[id]

        #mean_metric, std_metric, x = smoothing_mean_std(data, step_size=50)
        ax1.set_ylabel(id)
        label = labels[i] if labels else (id + str(i))
        x = list(range(len(DF)))

        ax1.plot(x,meandf[id], color=color, label="Mean")

        #ax1.plot(x,mindf[id], color=color, label="Min")
        #ax1.plot(x,maxdf[id], color=color, label="Max")

        ax1.fill_between(
            x,
            mindf[id],
            maxdf[id],
            alpha=0.2,
            color=color,
        )

    plt.legend(loc="best")
    fig.tight_layout()
    name = "_".join(ids)

    plt.savefig(os.path.join(path, name + "_mult_runs.png"), dpi=1200)

def plot_arr(arr, path, name): 
    fig, ax1 = plt.subplots()
    ax1.plot(arr)
    plt.savefig(os.path.join(path, name), dpi=1200)
