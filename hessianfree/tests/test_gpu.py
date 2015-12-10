import hessianfree as hf
import numpy as np
import pytest

if hf.gpu_enabled:
    from pycuda import gpuarray

pytestmark = [pytest.mark.skipif(not hf.gpu_enabled,
                                 reason="GPU packages not installed"),
              pytest.mark.parametrize("dtype", [np.float32, np.float64])]


@pytest.mark.parametrize("dot_type", [hf.gpu.kernel_wrappers.cublas_dot,
                                      hf.gpu.kernel_wrappers.shared_dot])
def test_dot(dtype, dot_type):
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

        a_gpu = gpuarray.to_gpu(a.astype(dtype))
        b_gpu = gpuarray.to_gpu(b.astype(dtype))

        c_gpu = dot_type(a_gpu, b_gpu, transpose_a=transpose_a,
                         transpose_b=transpose_b).get()
        c = np.dot(a.T if transpose_a else a, b.T if transpose_b else b)

        assert np.allclose(c, c_gpu, atol=1e-5)


def test_J_dot(dtype):
    for _ in range(100):
        N = np.random.randint(1, 100)
        batches = np.random.randint(1, 100)
        a = np.random.randn(batches, N, N).astype(dtype)
        b = np.random.randn(batches, N).astype(dtype)
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


def test_sum_cols(dtype):
    for _ in range(100):
        N = 200
        a = np.random.randn(np.random.randint(1, N),
                            np.random.randint(1, N)).astype(dtype)
        a_gpu = gpuarray.to_gpu(a)

        out = hf.gpu.sum_cols(a_gpu).get()

        assert np.allclose(out, np.sum(a, axis=0), atol=1e-5)


def test_multiply(dtype):
    for _ in range(100):
        N = 200
        a = np.random.randn(np.random.randint(1, N),
                            np.random.randint(1, N)).astype(dtype)
        b = np.random.randn(*a.shape).astype(dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)

        out = hf.gpu.multiply(a_gpu, b_gpu).get()

        assert np.allclose(out, a * b, atol=1e-5)


def test_ff_calc_G(dtype):
    inputs = np.random.randn(1000, 1).astype(dtype)
    ff = hf.FFNet([1, 10, 1], debug=(dtype == np.float64), use_GPU=True)
    ff.cache_minibatch(inputs, inputs)

    v = np.random.randn(ff.W.size).astype(dtype)
    gpu_Gv = ff.GPU_calc_G(v)
    cpu_Gv = ff.calc_G(v)

    assert np.allclose(gpu_Gv, cpu_Gv, rtol=1e-4)


def test_rnn_calc_G(dtype):
    inputs = np.random.randn(1000, 10, 1).astype(dtype)
    rnn = hf.RNNet([1, 10, 1], debug=(dtype == np.float64), use_GPU=True)
    rnn.cache_minibatch(inputs, inputs)
    rnn.optimizer = hf.opt.HessianFree()

    v = np.random.randn(rnn.W.size).astype(dtype)
    gpu_Gv = rnn.GPU_calc_G(v)
    cpu_Gv = rnn.calc_G(v)

    assert np.allclose(gpu_Gv, cpu_Gv, rtol=1e-4)


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_gpu.py")
