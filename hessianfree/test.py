import pickle
import sys
import ast

import numpy as np
import matplotlib.pyplot as plt

from hessianfree import FFNet, RNNet
from hessianfree.nonlinearities import (Logistic, Tanh, Softmax, SoftLIF, ReLU,
                                        Continuous, Linear, Nonlinearity,
                                        Gaussian)
from hessianfree.optimizers import HessianFree, SGD
from hessianfree.loss_funcs import Sparse, SquaredError, CrossEntropy


def test_xor():
    """Run a basic xor training test."""

    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = FFNet([2, 5, 1], debug=True)

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=2),
                   max_epochs=40, plotting=True)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
#                    max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs, ff.W)[-1]
    for i in range(4):
        print "-" * 20
        print "input", inputs[i]
        print "target", targets[i]
        print "output", outputs[i]


def test_mnist(model_args=None, run_args=None):
    """Test on the MNIST (digit classification) dataset."""

    # download dataset at http://deeplearning.net/data/mnist/mnist.pkl.gz
    with open("mnist.pkl", "rb") as f:
        train, _, test = pickle.load(f)

    if model_args is None:
        ff = FFNet([28 * 28, 1024, 512, 256, 32, 10],
                   layers=[Linear()] + [ReLU()] * 4 + [Softmax()],
                   use_GPU=True, debug=False)
    else:
        ff = FFNet(**model_args)

    inputs = train[0]
    targets = np.zeros((inputs.shape[0], 10), dtype=np.float32)
    targets[np.arange(inputs.shape[0]), train[1]] = 0.9
    targets += 0.01

    tmp = np.zeros((test[0].shape[0], 10), dtype=np.float32)
    tmp[np.arange(test[0].shape[0]), test[1]] = 0.9
    tmp += 0.01
    test = (test[0], tmp)

    def test_err(outputs, targets):
        return np.mean(np.argmax(outputs, axis=-1) !=
                       np.argmax(targets, axis=-1))

    if run_args is None:
        ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=250,
                                                              init_damping=45),
                       batch_size=7500, test=test, max_epochs=1000,
                       plotting=True, test_err=test_err)
    else:
        CG_iter = run_args.pop("CG_iter", 250)
        init_damping = run_args.pop("init_damping", 1)
        ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter,
                                                              init_damping),
                       test=test, test_err=test_err,
                       **run_args)

    output = ff.forward(test[0], ff.W)[-1]
    class_err = np.mean(np.argmax(output, axis=1) !=
                        np.argmax(test[1], axis=1))
    print "classification error", class_err


def test_cifar():
    """Test on the CIFAR (image classification) data set.

    Note: this is a WIP."""

    # download dataset at http://www.cs.toronto.edu/~kriz/cifar.html
    train = [None, None]
    for i in range(1, 6):
        with open("cifar/data_batch_%s" % i, "rb") as f:
            batch = pickle.load(f)
        if i == 1:
            train[0] = batch["data"]
            train[1] = batch["labels"]
        else:
            train[0] = np.concatenate((train[0], batch["data"]), axis=0)
            train[1] = np.concatenate((train[1], batch["labels"]), axis=0)
    with open("cifar/test_batch", "rb") as f:
        batch = pickle.load(f)
    test = [None, None]
    test[0] = batch["data"]
    test[1] = batch["labels"]

    # take random patches from training set
    dim = 24
    tmp = np.zeros((train[0].shape[0], dim * dim * 3))
    for i, t in enumerate(train[0]):
        img = t.reshape(3, 32, 32)

#         plt.figure()
#         plt.imshow(img.swapaxes(0, 2), interpolation="none")
#         plt.show()

        x_offset = np.random.randint(32 - dim)
        y_offset = np.random.randint(32 - dim)
        img = img[:, x_offset:x_offset + dim, y_offset:y_offset + dim]

#         plt.figure()
#         plt.imshow(img.swapaxes(0, 2), interpolation="none")
#         plt.show()

        tmp[i] = np.ravel(img)
    train[0] = tmp

    # take centre patches from test set
    tmp = np.zeros((test[0].shape[0], dim * dim * 3))
    for i, t in enumerate(test[0]):
        img = t.reshape(3, 32, 32)
        offset = (32 - dim) / 2
        img = img[:, offset:offset + dim, offset:offset + dim]
        tmp[i] = np.ravel(img)
    test[0] = tmp

    train[0] = train[0].astype(np.float32)
    tmp = np.zeros((len(train[0]), 10), dtype=np.float32)
    tmp[np.arange(len(train[0])), train[1]] = 1.0
    train[1] = tmp

    test[0] = test[0].astype(np.float32)
    tmp = np.zeros((len(test[0]), 10), dtype=np.float32)
    tmp[np.arange(len(test[0])), test[1]] = 1.0
    test[1] = tmp

    ff = FFNet([dim * dim * 3, 1024, 512, 256, 32, 10],
               layers=[Linear()] + [Tanh()] * 4 + [Softmax()],
               loss_type=CrossEntropy(), use_GPU=True, debug=False,
               load_weights=None)

    ff.run_batches(train[0], train[1], optimizer=HessianFree(CG_iter=300),
                   batch_size=5000, test=test, max_epochs=1000, plotting=True)

    output = ff.forward(test[0], ff.W)[-1]
    class_err = np.mean(np.argmax(output, axis=1) !=
                        np.argmax(test[1], axis=1))
    print "classification error", class_err


def test_softlif():
    """Example of a network using SoftLIF neurons."""

    lifs = SoftLIF(sigma=1, tau_ref=0.002, tau_rc=0.02, amp=0.01)

    inputs = np.asarray([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]],
                        dtype=np.float32)
    targets = np.asarray([[0.1], [0.9], [0.9], [0.1]], dtype=np.float32)

    ff = FFNet([2, 10, 1], layers=lifs, debug=True)

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=50),
                   max_epochs=50, plotting=True)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
#                    max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs, ff.W)[-1]
    for i in range(4):
        print "-" * 20
        print "input", inputs[i]
        print "target", targets[i]
        print "output", outputs[i]


def test_crossentropy():
    """Example of a network using cross-entropy error."""

    inputs = np.asarray([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]],
                        dtype=np.float32)
    targets = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)

    ff = FFNet([2, 5, 2], layers=[Linear(), Tanh(), Softmax()],
               debug=True, loss_type=CrossEntropy())

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=2),
                   max_epochs=40, plotting=True)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
#                    max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs, ff.W)[-1]
    for i in range(4):
        print "-" * 20
        print "input", inputs[i]
        print "target", targets[i]
        print "output", outputs[i]


def test_skip():
    """Example of a network with non-standard connectivity between layers."""

    inputs = np.asarray([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]],
                        dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = FFNet([2, 5, 5, 1], layers=Tanh(), debug=True,
               conns={0: [1, 2], 1: [2, 3], 2: [3]})

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=2),
                   max_epochs=40, plotting=True)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
#                    max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs, ff.W)[-1]
    for i in range(4):
        print "-" * 20
        print "input", inputs[i]
        print "target", targets[i]
        print "output", outputs[i]


def test_sparsity():
    """Example of a network with a loss function imposing sparsity on the
    neural activities."""

    inputs = np.asarray([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]],
                        dtype=np.float32)
    targets = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)

    ff = FFNet([2, 8, 2], layers=[Linear(), Tanh(), Softmax()],
               debug=True, loss_type=Sparse(CrossEntropy(), 0.1, target=-1))

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=10),
                   max_epochs=100, plotting=True)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1.0),
#                    max_epochs=10000, plotting=True)

    output = ff.forward(inputs, ff.W)
    for i in range(4):
        print "-" * 20
        print "input", inputs[i]
        print "target", targets[i]
        print "output", output[-1][i]
        print "activity", np.mean(output[1][i])


def test_profile():
    """Run a profiler on the code (uses MNIST example)."""

    np.random.seed(0)
    import cProfile
    import pstats

    cProfile.run("test_mnist(None, {'max_epochs':5, 'plotting':False, "
                 "'batch_size':7500})",
                 "profilestats")
    p = pstats.Stats("profilestats")
    p.strip_dirs().sort_stats('time').print_stats(20)


def test_GPU():
    """Profile CPU vs GPU performance (can be used to adjust the
    threshold in FFNet.init_GPU)."""

    import time

    ff = FFNet([1, 1], use_GPU=True, debug=False)
    ff.GPU_activations = None
    gpu = ff.outer_sum
    cpu = lambda a, b: np.ravel(np.einsum("ij,ik", a, b))

    vec_size = 11
    batch_size = 11
    reps = 100
    times = np.zeros((vec_size, batch_size, 2))
    for n in range(vec_size):
        for b in range(batch_size):
            x = np.random.randn(2 ** b, 2 ** n).astype(np.float32)
            y = np.random.randn(2 ** b, 2 ** n).astype(np.float32)

            start = time.time()
            for _ in range(reps):
                _ = cpu(x, y)
            times[n, b, 0] = time.time() - start

            start = time.time()
            for _ in range(reps):
                _ = gpu(x, y)
            times[n, b, 1] = time.time() - start

            print "n", n, "b", b, "times", times[n, b]

    print times
    print times[..., 1] > times[..., 0]


def test_integrator():
    """Test for a recurrent network, implementing an integrator."""

    n_inputs = 15
    sig_len = 50
    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    test = (inputs, targets)

    rnn = RNNet(shape=[1, 10, 1], struc_damping=None,
                layers=[Linear(), Logistic(), Logistic()],
                debug=False)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    test=test, max_epochs=100, plotting=True)

    # using gradient descent (for comparison)
#     rnn.run_batches(inputs, targets, optimizer=SGD(l_rate=0.1),
#                     batch_size=None, test=test, max_epochs=10000,
#                     plotting=True)

    plt.figure()
    plt.plot(inputs.squeeze().T)
    plt.title("inputs")

    plt.figure()
    plt.plot(targets.squeeze().T)
    plt.title("targets")

    outputs = rnn.forward(inputs, rnn.W)[-1]
    plt.figure()
    plt.plot(outputs.squeeze().T)
    plt.title("outputs")

    plt.show()


def test_continuous():
    """Example of a network using the Continuous nonlinearity."""

    n_inputs = 10
    sig_len = 50
    nl = Continuous(Logistic(), tau=np.random.uniform(1, 10, size=10), dt=0.6)
    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    test = (inputs, targets)

    rnn = RNNet(shape=[1, 10, 1], struc_damping=None,
                layers=[Linear(), nl, Logistic()],
                debug=False)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    batch_size=None, test=test, max_epochs=100, plotting=True)

    # using gradient descent (for comparison)
#     rnn.run_batches(inputs, targets, optimizer=SGD(l_rate=0.1),
#                     batch_size=None, test=test, max_epochs=10000,
#                     plotting=True)

    plt.figure()
    plt.plot(inputs.squeeze().T)
    plt.title("inputs")

    plt.figure()
    plt.plot(targets.squeeze().T)
    plt.title("targets")

    outputs = rnn.forward(inputs, rnn.W)[-1]
    plt.figure()
    plt.plot(outputs.squeeze().T)
    plt.title("outputs")

    plt.show()


def test_plant():
    """Example of a network using a dynamic plant as the output layer."""

    n_inputs = 1000
    sig_len = 20

    class Plant(Nonlinearity):
        def __init__(self, A, B, targets, init_state):
            super(Plant, self).__init__(stateful=True)

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

            self.B_matrix, self.d_B_matrix = self.B(self.state)

            self.state = (np.dot(self.state, self.A) +
                          np.einsum("ij,ijk->ik", x, self.B_matrix))

            return self.state[:x.shape[0]]
            # note: generally x will be the same shape as state, this just
            # handles the case where we're passed a single item instead
            # of batch)

        def d_activation(self, x, a):
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

        def __call__(self, x):
            self.inputs = np.concatenate((self.inputs, self.state[:, None, :]),
                                         axis=1)
            return self.state

        def get_inputs(self):
            return self.inputs

        def get_targets(self):
            return self.targets

        def reset(self, init=None):
            self.act_count = 0
            self.d_act_count = 0
            self.state = (self.init_state.copy() if init is None else
                          init.copy())
#             self.state = self.init_state[
#                 np.random.choice(np.arange(len(self.init_state)),
#                                  size=self.shape[0], replace=False)]
            self.inputs = np.zeros((self.shape[0], 0, self.shape[-1]),
                                   dtype=np.float32)
            self.B_matrix = self.d_B_matrix = None

    targets = np.ones((n_inputs, sig_len, 2), dtype=np.float32)
    targets[:, :, 1] = 0
    targets[:, :-1, :] = np.nan

    A = [[1, 0],
         [0.2, 1]]

    def B(state):
        B = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
        B[:, 1, 1] = np.cos(state[:, 0])

        d_B = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
        d_B[:, 1, 1] = -np.sin(state[:, 0])

        return B, d_B

    init1 = np.random.uniform(-1, 1, size=(n_inputs, 2))
#     init2 = np.random.uniform(-1, 1, size=(n_inputs, 2))

    plant = Plant(A, B, targets, init1)

    # TODO: why is the test generalization so bad?
    test = None  # (Plant(A, B, targets, init2), None)

    rnn = RNNet(shape=[2, 10, 10, 2], struc_damping=None,
                layers=[Linear(), Tanh(), Tanh(), plant],
                debug=False,
                rec_layers=[False, True, True, False],
                conns={0: [1, 2], 1: [2], 2: [3]},
                W_init_params={"coeff": 0.01}, W_rec_params={"coeff": 0.01})

    rnn.run_batches(plant, None, optimizer=HessianFree(CG_iter=100,
                                                       init_damping=1),
                    batch_size=None, test=test, max_epochs=100, plotting=True)

    # using gradient descent (for comparison)
#     rnn.run_batches(plant, None, optimizer=SGD(l_rate=0.01),
#                     batch_size=None, test=test, max_epochs=10000,
#                     plotting=True)

    outputs = rnn.forward(plant, rnn.W)[-1]

    plt.figure()
    plt.plot(plant.get_inputs()[:, :, 0].squeeze().T)
    plt.plot(plant.get_inputs()[:, :, 1].squeeze().T)
    plt.title("inputs")

    plt.figure()
    plt.plot(plant.get_targets()[:, :, 0].squeeze().T)
    plt.plot(plant.get_targets()[:, :, 1].squeeze().T)
    plt.title("targets")

    plt.figure()
    plt.plot(outputs[:, :, 0].squeeze().T)
    plt.plot(outputs[:, :, 1].squeeze().T)
    plt.title("outputs")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_xor()
    else:
        if "test_%s" % sys.argv[1] in locals():
            locals()["test_%s" % sys.argv[1]](*[ast.literal_eval(a)
                                                for a in sys.argv[2:]])
        else:
            print "Unknown test function (%s)" % sys.argv[1]
