import matplotlib.pyplot as plt
import numpy as np

from hessianbackprop import HessianBackprop


filename = "HF_weights.pkl"
layers = [28 * 28, 1000, 500, 250, 30, 10]
weight_shape = (28, 28)
n_neurons = 9

bp = HessianBackprop(layers, load_weights=filename)

W, _ = bp.get_layer(bp.W, 0)

for l in range(len(layers) - 1):
    if l > 0:
        W = np.dot(W, bp.get_layer(bp.W, l)[0])

    plt.figure()
    for i in range(n_neurons):
        ax = plt.subplot(int(np.ceil(np.sqrt(n_neurons))), int(np.floor(np.sqrt(n_neurons))), i)
        ax.axison = False
        plt.imshow(np.reshape(W[:, i], weight_shape))

plt.show()
