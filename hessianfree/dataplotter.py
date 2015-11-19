"""Run this script to display the data output during a run."""

import pickle
import threading

import matplotlib.pyplot as plt
import numpy as np


def run(filename):
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


def run_thread(filename):
    p = threading.Thread(target=run, args=(filename,))
    p.daemon = True
    p.start()


if __name__ == "__main__":
    run("HF_plots.pkl")
