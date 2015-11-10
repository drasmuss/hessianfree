import numpy as np
import pytest


from hessianfree import FFNet
from hessianfree.nonlinearities import (Tanh, Softmax, SoftLIF, Linear)
from hessianfree.optimizers import HessianFree, SGD
from hessianfree.loss_funcs import (SquaredError, CrossEntropy, SparseL1,
                                    ClassificationError)
from hessianfree.tests import use_GPU

pytestmark = pytest.mark.parametrize("use_GPU", use_GPU)


def test_xor(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = FFNet([2, 5, 1], debug=True, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=2),
                   max_epochs=40)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_SGD(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = FFNet([2, 5, 1], debug=False, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
                   max_epochs=10000)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-3


def test_softlif(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0.1], [1], [1], [0.1]], dtype=np.float32)

    lifs = SoftLIF(sigma=1, tau_ref=0.002, tau_rc=0.02, amp=0.01)

    ff = FFNet([2, 10, 1], layers=lifs, debug=True, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=50),
                   max_epochs=50)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_crossentropy(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)

    ff = FFNet([2, 5, 4], layers=[Linear(), Tanh(), Softmax()],
               debug=True, loss_type=CrossEntropy(), use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=50),
                   max_epochs=100)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_testerr(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=np.float32)

    ff = FFNet([2, 5, 2], layers=[Linear(), Tanh(), Softmax()],
               debug=True, loss_type=CrossEntropy(), use_GPU=use_GPU)

    err = ClassificationError()

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=50),
                   max_epochs=100, test_err=err, target_err=-1)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-4

    print outputs[-1]

    assert err.batch_loss(outputs, targets) == 0.0


def test_connections(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = FFNet([2, 5, 5, 1], layers=Tanh(), debug=True,
               conns={0: [1, 2], 1: [3], 2: [3]}, use_GPU=use_GPU)

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=50),
                   max_epochs=50)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_sparsity(use_GPU):
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = FFNet([2, 8, 1], debug=True, use_GPU=use_GPU,
               loss_type=[SquaredError(), SparseL1(0.01, target=0)])

    ff.run_batches(inputs, targets, optimizer=HessianFree(CG_iter=50),
                   max_epochs=100)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-2

    assert np.mean(outputs[1]) < 0.1


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_ffnet.py")
