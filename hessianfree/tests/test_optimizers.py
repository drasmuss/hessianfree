import numpy as np
import pytest

import hessianfree as hf
from hessianfree.tests import use_GPU

pytestmark = pytest.mark.parametrize("use_GPU", use_GPU)


def test_ff_CG(use_GPU):
    rng = np.random.RandomState(0)
    inputs = rng.randn(100, 1).astype(np.float32)
    targets = rng.randn(100, 1).astype(np.float32)
    ff = hf.FFNet([1, 10, 1], debug=False, use_GPU=use_GPU, rng=rng)
    ff.optimizer = hf.opt.HessianFree()
    ff.cache_minibatch(inputs, targets)

    deltas = ff.optimizer.conjugate_gradient(np.zeros(ff.W.size,
                                                      dtype=np.float32),
                                             ff.calc_grad(), iters=20,
                                             printing=False)

    assert deltas[0][0] == 3
    assert np.allclose(
        deltas[0][1],
        [-0.01693734, 0.00465961, 0.00173045, -0.00414165, -0.03843474,
         0.00636764, 0.01423731, -0.00433618, -0.00335347, 0.00935241,
         0.01242893, -0.00339621, -0.00137015, 0.00311182, 0.02883433,
         - 0.00534688, -0.01032545, 0.00328636, 0.00244868, -0.00678817,
         - 0.02461342, -0.02293827, -0.00737021, -0.01145663, -0.0116213,
         - 0.03512985, -0.02004906, -0.02885171, -0.01596764, -0.02105034,
         - 0.03943678], atol=1e-5)


def test_rnn_CG(use_GPU):
    rng = np.random.RandomState(0)
    inputs = rng.randn(100, 10, 2).astype(np.float32)
    targets = rng.randn(100, 10, 1).astype(np.float32)
    rnn = hf.RNNet([2, 5, 1], debug=False, use_GPU=use_GPU, rng=rng)
    rnn.optimizer = hf.opt.HessianFree()
    rnn.cache_minibatch(inputs, targets)

    deltas = rnn.optimizer.conjugate_gradient(np.zeros(rnn.W.size,
                                                       dtype=np.float32),
                                              rnn.calc_grad(), iters=20,
                                              printing=False)

    assert deltas[1][0] == 6
    assert np.allclose(
        deltas[1][1],
        [2.88910931e-03, -1.08404364e-02, 6.17342826e-04,
         - 1.85968506e-03, 1.71574634e-02, 3.08436429e-04,
         - 5.35693355e-02, -2.39962409e-03, 5.33994753e-03,
         3.52956937e-03, 1.83414537e-02, -1.20746918e-01,
         4.14435379e-03, 5.21760620e-03, 7.41007701e-02,
         - 2.86964715e-01, -2.21885830e-01, -3.84823292e-01,
         - 2.63742000e-01, -9.64779630e-02, -4.55241114e-01,
         9.68043320e-03, -5.81301711e-02, 1.87756377e-03,
         3.52657953e-05, 3.19301970e-02, 7.79627683e-03,
         - 4.76030372e-02, 1.58238632e-03, 1.87149423e-03,
         2.43508108e-02, 1.32407937e-02, -8.43726397e-02,
         2.58994917e-03, 2.43114564e-03, 4.95423339e-02,
         1.13963615e-02, -7.54035711e-02, 2.11156602e-03,
         4.81781084e-03, 4.49908487e-02, 4.63910261e-03,
         - 3.11208423e-02, 1.24892767e-03, 2.63486174e-03,
         1.77674163e-02, 1.60023139e-03, -1.40727460e-02,
         7.28542393e-04, 6.10395044e-04, 1.20819537e-02], atol=1e-5)

if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_optimizers.py")
