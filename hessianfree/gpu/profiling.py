import ast
import pstats
import sys
import time
from cProfile import Profile

import numpy as np
import pycuda
import pycuda.autoinit
from pycuda import gpuarray

from hessianfree import FFNet
from hessianfree.gpu import m_dot, simple_m_dot, outer_sum


def theshold_outer_sum():
    """Profile CPU vs GPU performance (can be used to adjust
    FFNet.GPU_threshold)."""

    ff = FFNet([1, 1], use_GPU=True, debug=False)

    # we always want to run on GPU
    ff.GPU_threshold = 0

    gpu = ff.outer_sum
    cpu = ff._outer_sum

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
            ff.GPU_activations = [gpuarray.to_gpu(x)]
            for _ in range(reps):
                _ = gpu([0], y)
            times[n, b, 1] = time.time() - start

            print "n", n, "b", b, "times", times[n, b]

    print times
    print times[..., 1] < times[..., 0]


def threshold_calc_G():
    vec_size = 11
    batch_size = 11
    reps = 100

    times = np.zeros((vec_size, batch_size, 3))
    for n in range(vec_size):
        for b in range(batch_size):
            inputs = np.random.randn(2 ** b, 1).astype(np.float32)
            targets = np.random.randn(2 ** b, 1).astype(np.float32)

            ff = FFNet([1, 2 ** n, 2 ** n, 1], use_GPU=False)
            ff.cache_minibatch(inputs, targets)

            v = np.random.randn(ff.W.size).astype(np.float32)

            start = time.time()
            for _ in range(reps):
                ff.calc_G(v)
            times[n, b, 0] = time.time() - start

            ff = FFNet([1, 2 ** n, 2 ** n, 1], use_GPU=True)
            ff.cache_minibatch(inputs, targets)

            start = time.time()
            for _ in range(reps):
                ff.calc_G(v)
            times[n, b, 1] = time.time() - start

            start = time.time()
            for _ in range(reps):
                ff.GPU_calc_G(v)
            times[n, b, 2] = time.time() - start

            print "n", n, "b", b, "times", times[n, b]

    print times
    print times[..., 1] < times[..., 0]


def profile_calc_G(cprofile=True):
    inputs = np.random.randn(1024, 1).astype(np.float32)
    targets = np.random.randn(1024, 1).astype(np.float32)
    N = 1024

    ff = FFNet([1, N, N, 1], use_GPU=True)
    ff.cache_minibatch(inputs, targets)

    v = np.random.randn(ff.W.size).astype(np.float32)

    for _ in range(5):
        # run it a few times to get rid of any startup overhead
        ff.GPU_calc_G(v)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.driver.start_profiler()

    for _ in range(100):
        _ = ff.calc_G(v)

#     for i in range(100):
#         Gv = gpuarray.to_gpu(tmp3)
#         for _ in range(10):
#             tmp = ff.m_dot(tmp4, tmp4)
#         Gv[:N * N] = tmp.ravel()
#         a = Gv.get()

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


def profile_m_dot(cprofile=True):
#     pycuda.compiler.DEFAULT_NVCC_FLAGS += ["-use_fast_math"]

    N = 1024
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    c = np.zeros((N, N), dtype=np.float32)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.zeros((N, N), np.float32)

    for _ in range(2):
        # run it a few times to get rid of any startup overhead
        m_dot(a_gpu, b_gpu, out=c_gpu)
        simple_m_dot(a_gpu, b_gpu, out=c_gpu)
        outer_sum(a_gpu, b_gpu, out=c_gpu)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.autoinit.context.synchronize()
        pycuda.driver.start_profiler()

    for _ in range(100):
#        np.dot(a, b, out=c)
#        simple_m_dot(a_gpu, b_gpu, out=c_gpu)
        m_dot(a_gpu, b_gpu, out=c_gpu, transpose_a=True, transpose_b=True)
#        outer_sum(a_gpu, b_gpu, out=c_gpu)
    c_gpu.get()

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()

if __name__ == "__main__":
    if sys.argv[1] in locals():
        locals()[sys.argv[1]](*[ast.literal_eval(a) for a in sys.argv[2:]])
    else:
        print "Unknown profile function (%s)" % sys.argv[1]
