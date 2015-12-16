import ast
import pstats
import sys
import time
from cProfile import Profile

import numpy as np
import pycuda
import pycuda.autoinit
from pycuda import gpuarray

import hessianfree as hf


def threshold_calc_G():
    """Compare GPU vs CPU performance on feedforward curvature calculation.

    This can use this to determine whether it is better to run some target
    network on the CPU or GPU."""

    batch_size = range(256, 1025, 256)
    layer_size = [1] + range(64, 513, 64)
    reps = 100

    times = np.zeros((len(batch_size), len(layer_size), 2))
    for i, b in enumerate(batch_size):
        inputs = np.random.randn(b, 1).astype(np.float32)
        targets = np.random.randn(b, 1).astype(np.float32)

        for j, n in enumerate(layer_size):
            ff = hf.FFNet([1, n, n, 1], use_GPU=False)
            ff.cache_minibatch(inputs, targets)

            v = np.random.randn(ff.W.size).astype(np.float32)

            for _ in range(5):
                ff.calc_G(v)

            start = time.time()
            for _ in range(reps):
                ff.calc_G(v)
            times[i, j, 0] = time.time() - start

            ff = hf.FFNet([1, n, n, 1], use_GPU=True)
            ff.cache_minibatch(inputs, targets)

            v = gpuarray.to_gpu(v)

            for _ in range(5):
                ff.GPU_calc_G(v)

            start = time.time()
            for _ in range(reps):
                ff.GPU_calc_G(v)

            v = v.get()
            times[i, j, 1] = time.time() - start

            print "b", b, "n", n, "times", times[i, j]

    print times[..., 1] - times[..., 0]

    print "batch size (%s) vs layer size (%s)" % (batch_size, layer_size)
    print " (True indicates GPU is faster)"
    print times[..., 1] < times[..., 0]


def threshold_rnn_calc_G():
    """Compare GPU vs CPU performance on recurrent curvature calculation.

    This can use this to determine whether it is better to run some target
    network on the CPU or GPU."""

    batch_size = 1024
    layer_size = [1] + range(32, 129, 32)
    sig_len = [1] + range(8, 33, 8)
    reps = 100

    times = np.zeros((len(sig_len), len(layer_size), 2))
    for i, b in enumerate(sig_len):
        inputs = np.random.randn(batch_size, b, 1).astype(np.float32)
        targets = np.random.randn(batch_size, b, 1).astype(np.float32)

        for j, n in enumerate(layer_size):
            rnn = hf.RNNet([1, n, 1], use_GPU=False)
            rnn.cache_minibatch(inputs, targets)

            v = np.random.randn(rnn.W.size).astype(np.float32)

            for _ in range(5):
                rnn.calc_G(v)

            start = time.time()
            for _ in range(reps):
                rnn.calc_G(v)
            times[i, j, 0] = time.time() - start

            rnn = hf.RNNet([1, n, 1], use_GPU=True)
            rnn.cache_minibatch(inputs, targets)

            v = gpuarray.to_gpu(v)

            for _ in range(5):
                rnn.GPU_calc_G(v)

            start = time.time()
            for _ in range(reps):
                rnn.GPU_calc_G(v)

            v = v.get()
            times[i, j, 1] = time.time() - start

            print "b", b, "n", n, "times", times[i, j]

    print times[..., 1] - times[..., 0]

    print "signal length (%s) versus layer size (%s)" % (sig_len, layer_size)
    print " (True indicates GPU is faster)"
    print times[..., 1] < times[..., 0]


def profile_calc_G(cprofile=True):
    """Run a profiler on the feedforward curvature calculation.

    :param bool cprofile: use True if profiling on the CPU, False if using the
        CUDA profiler
    """

    inputs = np.random.randn(1024, 1).astype(np.float32)
    targets = np.random.randn(1024, 1).astype(np.float32)
    N = 1024

    ff = hf.FFNet([1, N, N, 1], use_GPU=True)
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

    for _ in range(500):
        _ = ff.GPU_calc_G(v)

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


def profile_rnn_calc_G(cprofile=True):
    """Run a profiler on the recurrent curvature calculation.

    :param bool cprofile: use True if profiling on the CPU, False if using the
        CUDA profiler
    """

    inputs = np.random.randn(1024, 128, 1).astype(np.float32)
    targets = np.random.randn(1024, 128, 1).astype(np.float32)
    N = 128

    rnn = hf.RNNet([1, N, 1], use_GPU=True)
    rnn.optimizer = hf.opt.HessianFree()  # for struc_damping check
    rnn.cache_minibatch(inputs, targets)

    v = np.random.randn(rnn.W.size).astype(np.float32)

    for _ in range(2):
        # run it a few times to get rid of any startup overhead
        rnn.GPU_calc_G(v)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.driver.start_profiler()

    for _ in range(100):
        _ = rnn.GPU_calc_G(v)

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


def profile_dot(cprofile=True):
    """Run a profiler on the matrix multiplication kernel.

    :param bool cprofile: use True if profiling on the CPU, False if using the
        CUDA profiler
    """
    N = 1024
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.zeros((N, N), np.float32)

    for _ in range(2):
        # run it a few times to get rid of any startup overhead
        hf.gpu.dot(a_gpu, b_gpu, out=c_gpu)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.autoinit.context.synchronize()
        pycuda.driver.start_profiler()

    for _ in range(100):
        hf.gpu.dot(a_gpu, b_gpu, out=c_gpu, transpose_a=True,
                   transpose_b=True)
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
