import hessianfree as hf
import numpy as np
import pytest

from hessianfree.tests import pycuda_installed
if pycuda_installed:
    from hessianfree.gpu import m_dot, mv_dot
    from pycuda import gpuarray
    from pycuda.autoinit import context

pytestmark = pytest.mark.skipif(not pycuda_installed,
                                reason="PyCUDA not installed")


def test_mv_dot():
    # make sure we aren't accidentally using local memory
    assert hf.gpu.m_dot_kernel[0][0].local_size_bytes == 0

    # make sure we aren't using too many registers
    assert hf.gpu.m_dot_kernel[0][0].num_regs < 63

    # TODO: there is a rare test failure here, some kind of race condition?
#     a = np.random.randn(10, 76)
#     b = np.random.randn(56, 76)
#     a_gpu = gpuarray.to_gpu(a.astype(np.float32))
#     b_gpu = gpuarray.to_gpu(b.astype(np.float32))
#     for _ in range(1000):
#         c_gpu = mv_dot(a_gpu, b_gpu, batch_a=False, batch_v=True,
#                        transpose_a=False, transpose_v=True).get()
#         c = np.dot(a, b.T)
#
#         failed = np.where(np.abs(c - c_gpu) > 1e-5)
#         print zip(failed[0], failed[1])
#         print c[failed], c_gpu[failed]
#         assert np.allclose(c, c_gpu, atol=1e-5)

    for _ in range(1000):
        min_N = 1
        max_N = 100
        batch_a = np.random.choice([True, False])
        batch_b = np.random.choice([True, False])
        transpose_a = np.random.choice([True, False])
        transpose_b = np.random.choice([batch_b, batch_a and batch_b])
        if batch_a:
            a = np.random.randn(np.random.randint(min_N, max_N),
                                np.random.randint(min_N, max_N),
                                np.random.randint(min_N, max_N))
            if batch_b:
                b = np.random.randn(a.shape[0], a.shape[2 - transpose_a])
            else:
                b = np.random.randn(a.shape[2 - transpose_a], 1)

        else:
            a = np.random.randn(np.random.randint(min_N, max_N),
                                np.random.randint(min_N, max_N))

            if batch_b:
                b = np.random.randn(a.shape[1 - transpose_a],
                                    np.random.randint(min_N, max_N))
                if transpose_b:
                    b = np.ascontiguousarray(b.T)
            else:
                b = np.random.randn(a.shape[1 - transpose_a], 1)

        a_gpu = gpuarray.to_gpu(a.astype(np.float32))
        b_gpu = gpuarray.to_gpu(b.astype(np.float32))

#         print "test input", transpose_a, transpose_b, batch_a, batch_b

        c_gpu = mv_dot(a_gpu, b_gpu, batch_a=batch_a, batch_v=batch_b,
                       transpose_a=transpose_a, transpose_v=transpose_b).get()
        if batch_a and batch_b:
            c = np.einsum("ijk,ik->ij",
                          a.transpose((0, 2, 1)) if transpose_a else a, b)
        elif batch_a:
            c = np.einsum("ijk,k->ij",
                          a.transpose((0, 2, 1)) if transpose_a else a,
                          b.squeeze(axis=1))
        else:
            c = np.dot(a.T if transpose_a else a, b.T if transpose_b else b)

        print a.shape, b.shape, c.shape
        print transpose_a, transpose_b, batch_a, batch_b
        print np.max(np.abs(c - c_gpu))

        assert np.allclose(c, c_gpu, atol=1e-5)

        del a_gpu
        del b_gpu
        del c_gpu
        context.synchronize()


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

    ff.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=2),
                   max_epochs=40)

    # using gradient descent (for comparison)
#     ff.run_batches(inputs, targets, optimizer=SGD(l_rate=1),
#                    max_epochs=10000, plotting=True)

    outputs = ff.forward(inputs, ff.W)

    assert ff.loss.batch_loss(outputs, targets) < 1e-5


def test_J_dot():
    N = 65
    b = 5
    a = np.random.randn(b, N, N).astype(np.float32)
    b = np.random.randn(b, N).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

    out_gpu = hf.gpu.J_dot(a_gpu, b_gpu).get()
    out_cpu = np.einsum("ijk,ik->ij", a, b)

    assert np.allclose(out_gpu, out_cpu, atol=1e-5)

    out_gpu = hf.gpu.J_dot(a_gpu, b_gpu, transpose_J=True).get()
    out_cpu = np.einsum("ijk,ik->ij", a.transpose((0, 2, 1)), b)

    assert np.allclose(out_gpu, out_cpu, atol=1e-5)

    out_gpu = hf.gpu.J_dot(a_gpu, b_gpu, out=b_gpu).get()
    out_cpu = np.einsum("ijk,ik->ij", a, b)

    assert np.allclose(out_gpu, out_cpu, atol=1e-5)


def test_sum_cols():
    for _ in range(10):
        N = 5
        a = np.random.randn(np.random.randint(1, N),
                            np.random.randint(1, N)).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        out = hf.gpu.sum_cols(a_gpu).get()

#         print "result"
#         print np.sum(a, axis=0)
#         print out

        assert np.allclose(out, np.sum(a, axis=0), atol=1e-5)


def test_rnnet():
    n_inputs = 5
    sig_len = 10

    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    rnn = hf.RNNet(shape=[1, 5, 1], use_GPU=True, debug=False)

    rnn.run_batches(inputs, targets, optimizer=hf.opt.HessianFree(CG_iter=100),
                    max_epochs=30)

    outputs = rnn.forward(inputs, rnn.W)

    assert rnn.loss.batch_loss(outputs, targets) < 1e-4


def test_ff_calc_G():
    inputs = np.random.randn(2, 1).astype(np.float32)
    ff = hf.FFNet([1, 5, 1], debug=False, use_GPU=True)
    ff.cache_minibatch(inputs, inputs)

    v = np.random.randn(ff.W.size).astype(np.float32)
    gpu_Gv = ff.GPU_calc_G(v)
    cpu_Gv = ff.calc_G(v)

    assert np.allclose(gpu_Gv, cpu_Gv, rtol=1e-4)


def test_rnn_calc_G():
    inputs = np.random.randn(2, 3, 1).astype(np.float32)
    rnn = hf.RNNet([1, 2, 1], debug=False, use_GPU=True)
    rnn.cache_minibatch(inputs, inputs)
    rnn.optimizer = hf.opt.HessianFree()

    v = np.random.randn(rnn.W.size).astype(np.float32)
    gpu_Gv = rnn.GPU_calc_G(v)
    cpu_Gv = rnn.calc_G(v)

    assert np.allclose(gpu_Gv, cpu_Gv, rtol=1e-4)


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_gpu.py")
