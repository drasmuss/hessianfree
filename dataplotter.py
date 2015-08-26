"""Run this script to display the data output during a run."""

import pickle

import matplotlib.pyplot as plt
import numpy as np

filename = "HF_plots.pkl"
with open(filename, "rb") as f:
    plots = pickle.load(f)

axes = []
lines = []
for p in plots:
    plt.figure()
    plt.title(p)
    axes += [plt.gca()]
    lines += [plt.plot(plots[p])[0]]

while True:
    with open(filename, "rb") as f:
        plots = pickle.load(f)

    for i, p in enumerate(plots):
        lines[i].set_data(np.arange(len(plots[p])), plots[p])
        axes[i].relim()
        axes[i].autoscale_view()

    plt.draw()
    plt.pause(10)
