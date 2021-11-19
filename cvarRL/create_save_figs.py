import gym_minigrid
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm, Normalize
import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set_theme()
import matplotlib.pyplot as plt
from io import StringIO


import pickle as pkl
import gzip
import os
from io import StringIO
import pandas as pd

width = 12 - 2
height = 9 - 2

seed = list(range(1, 11))

budget = [
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
    1,
    25,
    100,
]
stochasticity = ["0.05"] * len(budget)
cost = ["0.7"] * len(budget)

nb = 10
fig, axn = plt.subplots(nrows=nb, ncols=3, figsize=(50, 50), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0, hspace=0.15)

final_map = [np.zeros((height, width))] * 3
perturbation_map = [np.zeros((height, width))] * 3

for idx, ax in enumerate(axn.flat):

    try:

        if idx in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:

            with gzip.open(
                os.path.join(
                    "results",
                    "test_experiment2_{}_{}_{}_{}_{}_5m_1.pkl.gz".format(
                        budget[idx], stochasticity[idx], cost[idx], seed[idx], 0
                    ),
                ),
                "rb",
            ) as f:
                map = pkl.load(f)

        else:

            with gzip.open(
                os.path.join(
                    "results",
                    "test_experiment3_{}_{}_{}_{}_{}_5m_1.pkl.gz".format(
                        budget[idx], stochasticity[idx], cost[idx], seed[idx], 0
                    ),
                ),
                "rb",
            ) as f:
                map = pkl.load(f)

    except FileNotFoundError:
        map = np.zeros((width, height))

    annotations = pd.DataFrame(np.zeros(map.T.shape))
    annotations[annotations == 0] = " "
    annotations = annotations.astype(str)
    annotations.loc[0, 9] = "G"
    annotations.loc[0, 0] = "S"

    if idx in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:
        final_map[0] = final_map[0] + map.T
    elif idx in [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]:
        final_map[1] = final_map[1] + map.T
    elif idx in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]:
        final_map[2] = final_map[2] + map.T

    sns.heatmap(map.T, linewidths=0.5, ax=ax, cmap="YlGnBu", annot=annotations, fmt="s", annot_kws={"fontsize": 20})
    if idx in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:
        ax.set_title("No antagonist".format(budget[idx]), fontsize=25)
    else:
        ax.set_title("Antagonist with budget $\eta = {}$".format(budget[idx]), fontsize=25)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

fig.savefig("results/all_seeds.png")


def nans(carte):
    for i in [0, 1]:
        for j in [2, 3, 4, 5, 6, 7]:
            carte[i, j] = np.nan
    return carte


fig, axn = plt.subplots(nrows=1, ncols=3, figsize=(40, 10), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0, hspace=0.15)

maxes = []
budget = [1, 25, 100]
letter = ["a", "b", "c"]
for idx, ax in enumerate(axn.flat):

    policy = final_map[idx] / nb
    maxes.append(np.nanmax(policy))
    annotations = pd.DataFrame(np.round(policy, 2))
    annotations = annotations.where(annotations < 0, "")

    annotations = annotations.astype(str)
    annotations.loc[0, 9] = "G"
    annotations.loc[0, 0] = "S"

    final_map[idx] = nans(final_map[idx])
    sns.heatmap(
        final_map[idx] / nb,
        linewidths=0.5,
        ax=ax,
        cmap="YlGnBu",
        annot=annotations,
        annot_kws={"fontsize": 20},
        fmt="s",
        vmin=0,
        vmax=1.3,
    )
    ax.set_facecolor("xkcd:salmon")

    if idx == 0:
        ax.set_title(" a) No antagonist", fontsize=25)
    else:
        ax.set_title("{}) Antagonist budget $\eta ={}$".format(letter[idx], budget[idx]), fontsize=25)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

fig.savefig("results/averaged_seeds.png")


def nans(carte):
    for i in [0, 1]:
        for j in [2, 3, 4, 5, 6, 7]:
            carte[i, j] = np.nan
    return carte


maxes = []
budget = [1, 25, 100]
letter = ["a", "b", "c"]
for idx, ax in enumerate(budget):

    fig, axn = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), sharex=True, sharey=True)

    policy = final_map[idx] / nb
    maxes.append(np.nanmax(policy))
    annotations = pd.DataFrame(np.round(policy, 2))
    annotations = annotations.where(annotations < 0, "")

    annotations = annotations.astype(str)
    annotations.loc[0, 9] = "G"
    annotations.loc[0, 0] = "S"

    final_map[idx] = nans(final_map[idx])
    a = sns.heatmap(
        final_map[idx] / nb,
        linewidths=0.5,
        cmap="YlGnBu",
        annot=annotations,
        annot_kws={"fontsize": 20},
        fmt="s",
        vmin=0,
        vmax=1.3,
    )
    a.set_facecolor("xkcd:salmon")

    fig.savefig("results/averaged_seeds_{}.png".format(idx), bbox_inches="tight")
