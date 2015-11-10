import hessianfree as hf
import numpy as np
import pytest

from hessianfree.tests import pycuda_installed
if pycuda_installed:
    from hessianfree.gpu import m_dot
    from pycuda import gpuarray

pytestmark = pytest.mark.skipif(not pycuda_installed,
                                reason="PyCUDA not installed")


def test_m_dot():
    # make sure we aren't accidentally using local memory
    assert hf.gpu.m_dot_kernel[0][0].local_size_bytes == 0

    # make sure we aren't using too many registers
    assert hf.gpu.m_dot_kernel[0][0].num_regs < 63

    for _ in range(1000):
        N = 100
        a = np.random.randn(np.random.randint(1, N),
                            np.random.randint(1, N))
        transpose_a = np.random.choice([True, False])
        transpose_b = np.random.choice([True, False])
        if transpose_b & transpose_a:
            b = np.random.randn(np.random.randint(1, N), a.shape[0])
        elif transpose_b:
            b = np.random.randn(np.random.randint(1, N), a.shape[1])
        elif transpose_a:
            b = np.random.randn(a.shape[0], np.random.randint(1, N))
        else:
            b = np.random.randn(a.shape[1], np.random.randint(1, N))

        a_gpu = gpuarray.to_gpu(a.astype(np.float32))
        b_gpu = gpuarray.to_gpu(b.astype(np.float32))

        c_gpu = m_dot(a_gpu, b_gpu, transpose_a=transpose_a,
                      transpose_b=transpose_b).get()
        c = np.dot(a.T if transpose_a else a, b.T if transpose_b else b)

#         print c
#         print c_gpu

        assert np.allclose(c, c_gpu, atol=1e-5)


def test_ffnet():
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    ff = hf.FFNet([2, 5, 1], debug=True, use_GPU=True)
    ff.GPU_threshold = 0

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=2),
                   max_epochs=40)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
#                    max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_J_dot():
    N = 100
    b = 1000
    a = np.random.randn(b, N, N).astype(np.float32)
    b = np.random.randn(b, N).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

    out_gpu = hf.gpu.J_dot(a_gpu, b_gpu).get()
    out_cpu = np.einsum("ijk,ik->ij", a, b)

    assert np.allclose(out_gpu, out_cpu, atol=1e-5)

    out_gpu = hf.gpu.J_dot(a_gpu, b_gpu, transpose_a=True).get()
    out_cpu = np.einsum("ijk,ik->ij", a.transpose((0, 2, 1)), b)

    assert np.allclose(out_gpu, out_cpu, atol=1e-5)

    out_gpu = hf.gpu.J_dot(a_gpu, b_gpu, out=b_gpu).get()
    out_cpu = np.einsum("ijk,ik->ij", a, b)

    assert np.allclose(out_gpu, out_cpu, atol=1e-5)


# def test_rnnet():
#     n_inputs = 5
#     sig_len = 10
#
#     inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
#                       np.ones(sig_len))[:, :, None]
#     targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
#                        np.linspace(0, 1, sig_len))[:, :, None]
#     inputs = inputs.astype(np.float32)
#     targets = targets.astype(np.float32)
#
#     rnn = hf.RNNet(shape=[1, 10, 1], use_GPU=True, debug=True)
#
#     rnn.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=100),
#                     max_epochs=30)
#
#     outputs = rnn.forward(inputs, rnn.W)
#
#     assert rnn.loss.batch_loss(outputs, targets) < 1e-4


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_gpu.py")
