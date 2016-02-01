import pytest
import numpy as np
import matplotlib.pyplot as plt

import hessianfree as hf
from hessianfree.nonlinearities import Logistic, Continuous, Tanh, Linear
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

    rnn = hf.RNNet(shape=[1, 5, 1], debug=True, use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30, print_period=None)

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

    optimizer = HessianFree(CG_iter=100)

    rnn = hf.RNNet(
        shape=[1, 5, 1],
        loss_type=[hf.loss_funcs.SquaredError(),
                   hf.loss_funcs.StructuralDamping(0.1, optimizer=optimizer)],
        debug=True, use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=optimizer,
                    max_epochs=30, print_period=None)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


def test_continuous(use_GPU):
    n_inputs = 3
    sig_len = 5
    nl = Continuous(Logistic(), tau=np.random.uniform(1, 3, size=5), dt=0.9)
    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    rnn = hf.RNNet(shape=[1, 5, 1], layers=[Linear(), nl, Logistic()],
                   debug=True, use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30, print_period=None)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


def test_asym_dact(use_GPU):
    class Roll(hf.nl.Nonlinearity):
        def activation(self, x):
            return np.roll(x, 1, axis=-1)

        def d_activation(self, x, _):
            d_act = np.roll(np.eye(x.shape[-1], dtype=x.dtype), 1, axis=0)
            return np.resize(d_act, np.concatenate((x.shape[:-1],
                                                    d_act.shape)))

    n_inputs = 3
    sig_len = 5

    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    rnn = hf.RNNet(shape=[1, 5, 1], layers=Roll(), debug=True, use_GPU=use_GPU)

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30, print_period=None)


def test_plant(use_GPU):
    n_inputs = 32
    sig_len = 15

    class Plant(hf.nl.Plant):
        # this plant implements a simple dynamic system, with two-dimensional
        # state representing [position, velocity]
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

    # initial position
    init_state = np.zeros((n_inputs, 2))
    init_state[:, 0] = np.linspace(-1, 1, n_inputs)

    # the target will be to end at position 1 with velocity 0
    targets = np.ones((n_inputs, sig_len, 2), dtype=np.float32)
    targets[:, :, 1] = 0
    targets[:, :-1, :] = np.nan

    plant = Plant(A, B, targets, init_state)

    rnn = hf.RNNet(shape=[2, 16, 2], layers=[Linear(), Tanh(), plant],
                   W_init_params={"coeff": 0.1}, W_rec_params={"coeff": 0.1},
                   use_GPU=use_GPU, rng=np.random.RandomState(0), debug=False)

    rnn.run_batches(plant, None, HessianFree(CG_iter=20, init_damping=10),
                    max_epochs=150, plotting=True, print_period=None)

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

    rnn = hf.RNNet(shape=[1, 5, 1], debug=True, use_GPU=use_GPU,
                   truncation=(3, 3))

    rnn.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=100),
                    max_epochs=30, print_period=None)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_rnnet.py")
