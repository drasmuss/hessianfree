from __future__ import print_function

import ast
from cProfile import Profile
import pickle
import pstats
import sys

import numpy as np
import matplotlib.pyplot as plt

import hessianfree as hf


def xor(use_hf=True):
    """Run a basic xor training test.

    :param bool use_hf: if True run example using Hessian-free optimization,
        otherwise use stochastic gradient descent
    """

    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 1])

    if use_hf:
        ff.run_batches(inputs, targets,
                       optimizer=hf.opt.HessianFree(CG_iter=2),
                       max_epochs=40, plotting=True)
    else:
        # using gradient descent (for comparison)
        ff.run_batches(inputs, targets, optimizer=hf.opt.SGD(l_rate=1),
                       max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs)[-1]
    for i in range(4):
        print("-" * 2)
        print("input", inputs[i])
        print("target", targets[i])
        print("output", outputs[i])


def crossentropy():
    """A network that modifies the layer types and loss function."""

    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 2], layers=[hf.nl.Linear(), hf.nl.Tanh(),
                                     hf.nl.Softmax()],
                  loss_type=hf.loss_funcs.CrossEntropy())

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=2),
                   max_epochs=40, plotting=True)

    outputs = ff.forward(inputs)[-1]
    for i in range(4):
        print("-" * 20)
        print("input", inputs[i])
        print("target", targets[i])
        print("output", outputs[i])


def connections():
    """A network with non-standard connectivity between layers."""

    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 5, 1], layers=hf.nl.Tanh(),
                  conns={0: [1, 2], 1: [2, 3], 2: [3]})

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=2),
                   max_epochs=40, plotting=True)

    outputs = ff.forward(inputs)[-1]
    for i in range(4):
        print("-" * 20)
        print("input", inputs[i])
        print("target", targets[i])
        print("output", outputs[i])


def mnist(model_args=None, run_args=None):
    """Test on the MNIST (digit classification) dataset.

    Download dataset at http://deeplearning.net/data/mnist/mnist.pkl.gz

    :param dict model_args: kwargs that will be passed to the :class:`.FFNet`
        constructor
    :param dict run_args: kwargs that will be passed to :meth:`.run_batches`
    """

    with open("mnist.pkl", "rb") as f:
        train, _, test = pickle.load(f)

    if model_args is None:
        ff = hf.FFNet([28 * 28, 1024, 512, 256, 32, 10],
                      layers=([hf.nl.Linear()] + [hf.nl.ReLU()] * 4 +
                              [hf.nl.Softmax()]),
                      use_GPU=True, debug=False)
    else:
        ff = hf.FFNet([28 * 28, 1024, 512, 256, 32, 10],
                      layers=([hf.nl.Linear()] + [hf.nl.ReLU()] * 4 +
                              [hf.nl.Softmax()]),
                      **model_args)

    inputs = train[0]
    targets = np.zeros((inputs.shape[0], 10), dtype=np.float32)
    targets[np.arange(inputs.shape[0]), train[1]] = 0.9
    targets += 0.01

    tmp = np.zeros((test[0].shape[0], 10), dtype=np.float32)
    tmp[np.arange(test[0].shape[0]), test[1]] = 0.9
    tmp += 0.01
    test = (test[0], tmp)

    if run_args is None:
        ff.run_batches(inputs, targets,
                       optimizer=hf.opt.HessianFree(CG_iter=250,
                                                    init_damping=45),
                       batch_size=7500, test=test, max_epochs=1000,
                       test_err=hf.loss_funcs.ClassificationError(),
                       plotting=True)
    else:
        CG_iter = run_args.pop("CG_iter", 250)
        init_damping = run_args.pop("init_damping", 45)
        ff.run_batches(inputs, targets,
                       optimizer=hf.opt.HessianFree(CG_iter, init_damping),
                       test=test, test_err=hf.loss_funcs.ClassificationError(),
                       **run_args)

    output = ff.forward(test[0])
    print("classification error",
          hf.loss_funcs.ClassificationError().batch_loss(output, test[1]))


def integrator(model_args=None, run_args=None, n_inputs=15, sig_len=10,
               plots=True):
    """A recurrent network implementing an integrator.

    :param dict model_args: kwargs that will be passed to the :class:`.RNNet`
        constructor
    :param dict run_args: kwargs that will be passed to :meth:`.run_batches`
    :param int n_inputs: size of batch to train on
    :param int sig_len: number of timesteps to run for
    :param bool plots: display plots of trained output
    """

    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    test = (inputs, targets)

    if model_args is None:
        rnn = hf.RNNet(shape=[1, 10, 1], layers=hf.nl.Logistic(),
                       debug=False, use_GPU=False)
    else:
        rnn = hf.RNNet(**model_args)

    if run_args is None:
        rnn.run_batches(inputs, targets,
                        optimizer=hf.opt.HessianFree(CG_iter=100),
                        test=test, max_epochs=30, plotting=plots)
    else:
        CG_iter = run_args.pop("CG_iter", 100)
        init_damping = run_args.pop("init_damping", 1)
        rnn.run_batches(inputs, targets,
                        optimizer=hf.opt.HessianFree(CG_iter, init_damping),
                        test=test, plotting=plots, **run_args)

    if plots:
        plt.figure()
        plt.plot(inputs.squeeze().T)
        plt.title("inputs")

        plt.figure()
        plt.plot(targets.squeeze().T)
        plt.title("targets")

        outputs = rnn.forward(inputs)[-1]
        plt.figure()
        plt.plot(outputs.squeeze().T)
        plt.title("outputs")

        plt.show()


def adding(T=50, plots=True):
    """The canonical "adding" test of long-range dependency learning for RNNs.

    :param int T: length of the test signal
    :param bool plots: display plots of trained output
    """

    # set up inputs
    N = 100000
    test_cut = int(N * 0.9)

    vals = np.random.uniform(0, 1, size=(N, T, 1)).astype(np.float32)
    mask = np.zeros((N, T, 1), dtype=np.float32)
    for m in mask:
        m[np.random.randint(T / 10)] = 1
        m[np.random.randint(T / 10, T / 2)] = 1
    inputs = np.concatenate((vals, mask), axis=-1)

    tmp = np.zeros_like(vals)
    tmp[mask.astype(np.bool)] = vals[mask.astype(np.bool)]

    targets = np.zeros((N, T, 1), dtype=np.float32)
    targets[:] = np.nan
    targets[:, -1] = np.sum(tmp, axis=1, dtype=np.float32)

    test = (inputs[test_cut:], targets[test_cut:])

    # build network
    optimizer = hf.opt.HessianFree(CG_iter=60, init_damping=20)
    rnn = hf.RNNet(
        shape=[2, 32, 64, 1],
        layers=[hf.nl.Linear(), hf.nl.ReLU(),
                hf.nl.Continuous(hf.nl.ReLU(), tau=20), hf.nl.ReLU()],
        W_init_params={"coeff": 0.25},
        loss_type=[hf.loss_funcs.SquaredError(),
                   hf.loss_funcs.StructuralDamping(1e-4, layers=[2],
                                                   optimizer=optimizer)],
        rec_layers=[2], use_GPU=True, debug=False,
        rng=np.random.RandomState(0))

    # scale spectral radius of recurrent weights
    W, _ = rnn.get_weights(rnn.W, (2, 2))
    W *= 1.0 / np.max(np.abs(np.linalg.eigvals(W)))

    rnn.run_batches(inputs[:test_cut], targets[:test_cut],
                    optimizer=optimizer, batch_size=1024, test=test,
                    max_epochs=50, plotting=plots,
                    test_err=hf.loss_funcs.SquaredError())

    if plots:
        outputs = rnn.forward(inputs[:20])
        plt.figure()
        lines = plt.plot(outputs[-1][:].squeeze().T)
        plt.scatter(np.ones(outputs[-1].shape[0]) * outputs[-1].shape[1],
                    targets[:20, -1], c=[plt.getp(l, "color") for l in lines])
        plt.title("outputs")

        plt.show()


def plant(plots=True):
    """A network using a dynamic plant as the output layer.

    :param bool plots: display plots of trained output
    """

    n_inputs = 32
    sig_len = 15

    class Plant(hf.nl.Plant):
        # this plant implements a simple dynamic system, with two-dimensional
        # state representing [position, velocity]
        def __init__(self, A, B, targets, init_state):
            super(Plant, self).__init__()

            self.A = np.asarray(A)
            self.B = B

            self.targets = targets
            self.init_state = init_state

            self.shape = [n_inputs, sig_len, len(A)]

            # derivative of output with respect to state (constant, so just
            # compute it once here)
            self.d_output = np.resize(np.eye(self.shape[-1]),
                                      (n_inputs, self.shape[-1],
                                       self.shape[-1], 1))

            self.reset()

        def activation(self, x):
            self.act_count += 1

            # this implements a basic s_{t+1} = A*s_t + B*x dynamic system.
            # but to make things a little more complicated we allow the B
            # matrix to be dynamic, so it's actually
            # s_{t+1} = A*s_t + B(s_t)*x

            self.B_matrix, self.d_B_matrix = self.B(self.state)

            self.state = (np.dot(self.state, self.A) +
                          np.einsum("ij,ijk->ik", x, self.B_matrix))

            return self.state[:x.shape[0]]

        def d_activation(self, x, _):
            self.d_act_count += 1
            assert self.act_count == self.d_act_count

            # derivative of state with respect to input
            d_input = self.B_matrix.transpose((0, 2, 1))[..., None]

            # derivative of state with respect to previous state
            d_state = np.resize(self.A.T, np.concatenate(([x.shape[0]],
                                                          self.A.shape)))
            d_state[:, 1, 0] += x[:, 1] * self.d_B_matrix[:, 1, 1]
            d_state = d_state[..., None]

            return np.concatenate((d_input, d_state, self.d_output), axis=-1)

        def __call__(self, _):
            self.inputs = np.concatenate((self.inputs, self.state[:, None, :]),
                                         axis=1)
            return self.state

        def get_vecs(self):
            return self.inputs, self.targets

        def reset(self, init=None):
            self.act_count = 0
            self.d_act_count = 0
            self.state = (self.init_state.copy() if init is None else
                          init.copy())
            self.inputs = np.zeros((self.shape[0], 0, self.shape[-1]),
                                   dtype=np.float32)
            self.B_matrix = self.d_B_matrix = None

    # static A matrix (converts velocity into a change in position)
    A = [[1, 0],
         [0.2, 1]]

    # dynamic B(s) matrix (converts input into velocity, modulated by current
    # state)
    # note that this dynamic B matrix doesn't really make much sense, it's
    # just here to demonstrate what happens with a plant whose dynamics
    # change over time
    def B(state):
        B = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
        B[:, 1, 1] = np.tanh(state[:, 0])

        d_B = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
        d_B[:, 1, 1] = 1 - np.tanh(state[:, 0]) ** 2

        return B, d_B

    # random initial position and velocity
    init_state = np.random.uniform(-0.5, 0.5, size=(n_inputs, 2))

    # the target will be to end at position 1 with velocity 0
    targets = np.ones((n_inputs, sig_len, 2), dtype=np.float32)
    targets[:, :, 1] = 0
    targets[:, :-1, :] = np.nan

    plant = Plant(A, B, targets, init_state)

    rnn = hf.RNNet(shape=[2, 16, 2], layers=[hf.nl.Linear(), hf.nl.Tanh(),
                                             plant],
                   W_init_params={"coeff": 0.1}, W_rec_params={"coeff": 0.1},
                   rng=np.random.RandomState(0))

    rnn.run_batches(plant, None, hf.opt.HessianFree(CG_iter=20,
                                                    init_damping=10),
                    max_epochs=150, plotting=plots)

    # using gradient descent (for comparison)
#     rnn.run_batches(plant, None, optimizer=SGD(l_rate=0.01),
#                     batch_size=None, test=test, max_epochs=10000,
#                     plotting=True)

    if plots:
        outputs = rnn.forward(plant)[-1]

        plt.figure()
        plt.plot(outputs[:, :, 0].squeeze().T)
        plt.title("position")

        plt.figure()
        plt.plot(outputs[:, :, 1].squeeze().T)
        plt.title("velocity")

        plt.show()


def profile(func, max_epochs=15, use_GPU=False, cprofile=True):
    """Run a profiler on the code.

    :param str func: the demo function to be profiled (can be 'mnist' or
        'integrator')
    :param int max_epochs: maximum number of iterations to run
    :param bool use_GPU: run optimization on GPU
    :param bool cprofile: if True then run the profiling on the CPU, otherwise
        use CUDA profiler
    """

    if cprofile:
        p = Profile()
        p.enable()
    else:
        import pycuda
        pycuda.driver.start_profiler()

    if func == "mnist":
        mnist({'use_GPU': use_GPU, 'rng': np.random.RandomState(0)},
              {'max_epochs': max_epochs, 'plotting': False, 'batch_size': 7500,
               'CG_iter': 10})
    elif func == "integrator":
        integrator({'shape': [1, 100, 1], 'layers': hf.nl.Logistic(),
                    'use_GPU': use_GPU, 'debug': False,
                    'rng': np.random.RandomState(0)},
                   {'max_epochs': max_epochs, 'CG_iter': 10},
                   n_inputs=500, sig_len=200, plots=False)
    else:
        raise ValueError("Unknown profile function")

    if cprofile:
        p.disable()

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        xor()
    else:
        if sys.argv[1] in locals():
            locals()[sys.argv[1]](*[ast.literal_eval(a) for a in sys.argv[2:]])
        else:
            print("Unknown demo function (%s)" % sys.argv[1])
