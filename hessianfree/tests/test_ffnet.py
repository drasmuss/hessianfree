import numpy as np
import pytest

import hessianfree as hf
from hessianfree.tests import use_GPU

pytestmark = pytest.mark.parametrize("use_GPU", use_GPU)


def test_xor(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 1], debug=True, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=2),
                   max_epochs=40, print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_SGD(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 1], debug=False, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=hf.opt.SGD(l_rate=1),
                   max_epochs=10000, print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-3


def test_softlif(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0.1], [1], [1], [0.1]], dtype=np.float32)

    lifs = hf.nl.SoftLIF(sigma=1, tau_ref=0.002, tau_rc=0.02, amp=0.01)

    ff = hf.FFNet([2, 10, 1], layers=lifs, debug=True, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=50),
                   max_epochs=50, print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_crossentropy(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 4], layers=[hf.nl.Linear(), hf.nl.Tanh(),
                                     hf.nl.Softmax()],
                  debug=True, loss_type=hf.loss_funcs.CrossEntropy(),
                  use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=50),
                   max_epochs=100, print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_testerr(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 2], layers=[hf.nl.Linear(), hf.nl.Tanh(),
                                     hf.nl.Softmax()],
                  debug=True, loss_type=hf.loss_funcs.CrossEntropy(),
                  use_GPU=use_GPU)

    err = hf.loss_funcs.ClassificationError()

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=50),
                   max_epochs=100, test_err=err, target_err=-1,
                   print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-4

    print outputs[-1]

    assert err.batch_loss(outputs, targets) == 0.0


def test_connections(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 5, 1], layers=hf.nl.Tanh(), debug=True,
                  conns={0: [1, 2], 1: [3], 2: [3]}, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=50),
                   max_epochs=50, print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_sparsity(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 8, 1], debug=True, use_GPU=use_GPU,
                  loss_type=[hf.loss_funcs.SquaredError(),
                             hf.loss_funcs.SparseL2(0.01, target=0)])

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=50),
                   max_epochs=100, print_period=None)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-2

    assert np.mean(outputs[1]) < 0.1


def test_asym_dact(use_GPU):
    class Roll(hf.nl.Nonlinearity):
        def activation(self, x):
            return np.roll(x, 1, axis=-1)

        def d_activation(self, x, _):
            d_act = np.roll(np.eye(x.shape[-1], dtype=x.dtype), 1, axis=0)
            return np.resize(d_act, np.concatenate((x.shape[:-1],
                                                    d_act.shape)))

    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 1], layers=Roll(), debug=True, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=2),
                   max_epochs=40, print_period=None)

if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_ffnet.py")
