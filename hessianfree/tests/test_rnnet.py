import pytest
import numpy as np
import matplotlib.pyplot as plt

import hessianfree as hf
from hessianfree.nonlinearities import (Logistic, Continuous, Tanh, Linear,
                                        Nonlinearity)
from hessianfree.optimizers import HessianFree
from hessianfree.tests import use_GPU


pytestmark = pytest.mark.parametrize("use_GPU", use_GPU)


def test_integrator(use_GPU):
    n_inputs = 3
    sig_len = 5

    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    rnn = hf.RNNet(shape=[1, 10, 1], debug=True, use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


def test_strucdamping(use_GPU):
    n_inputs = 3
    sig_len = 5

    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    # TODO: run with debug=True when struc_damping check works
    rnn = hf.RNNet(shape=[1, 10, 1], debug=False, struc_damping=0.1,
                   use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


def test_continuous(use_GPU):
    n_inputs = 3
    sig_len = 5
    nl = Continuous(Logistic(), tau=np.random.uniform(1, 3, size=10), dt=0.9)
    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    rnn = hf.RNNet(shape=[1, 10, 1], layers=[Linear(), nl, Logistic()],
                   debug=True, use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


def test_plant(use_GPU):
    np.random.seed(0)
    n_inputs = 10
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

        def get_inputs(self):
            return self.inputs

        def get_targets(self):
            return self.targets

        def reset(self, init=None):
            self.act_count = 0
            self.d_act_count = 0
            self.state = (self.init_state.copy() if init is None else
                          init.copy())
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

    plant = Plant(A, B, targets, init1)

    rnn = hf.RNNet(shape=[2, 10, 10, 2], debug=False,
                   layers=[Linear(), Tanh(), Tanh(), plant],
                   conns={0: [1, 2], 1: [2], 2: [3]},
                   W_init_params={"coeff": 0.01}, W_rec_params={"coeff": 0.01},
                   use_GPU=use_GPU)
    rnn.run_batches(plant, None, optimizer=HessianFree(CG_iter=100),
                    max_epochs=100, plotting=False)

    outputs = rnn.forward(plant, rnn.W)

    try:
        assert rnn.loss.batch_loss(outputs, targets) < 1e-2
    except AssertionError:
        plt.figure()
        plt.plot(outputs[-1][:, :, 0].squeeze().T)
        plt.plot(outputs[-1][:, :, 1].squeeze().T)
        plt.title("outputs")
        plt.savefig("test_plant_outputs.png")

        raise


def test_truncation(use_GPU):
    n_inputs = 2
    sig_len = 6

    inputs = np.ones((n_inputs, sig_len, 1), dtype=np.float32) * 0.5
    targets = np.ones((n_inputs, sig_len, 1), dtype=np.float32) * 0.5

    rnn = hf.RNNet(shape=[1, 10, 1], debug=True, use_GPU=use_GPU,
                   truncation=(3, 3))

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_rnnet.py")
