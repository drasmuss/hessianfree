import copy
from functools import wraps

import numpy as np
from pycuda import gpuarray, driver
from skcuda import cublas, misc

import hessianfree as hf

misc.init()


def find_block_len(n_threads, threads_per, vec_len):
    # need to divide n_threads into blocks of size n*threads_per. we
    # want n to be as large as possible, so we use all the threads in
    # a block. but vec_len also needs to be evenly divisible by n (I
    # don't think it's possible to have different sized blocks in
    # different grid cells). so we want the highest factor of vec_len
    # that is <= n_threads/threads_per.
    start = int(n_threads / threads_per)

    if start >= vec_len:
        return np.asscalar(vec_len)

    mid = int(np.sqrt(vec_len))
    for n in range(start, 0 if start < mid else mid - 1, -1):
        if vec_len % n == 0:
            return n

    return 1


def debug_wrapper(cpu_func, debug=False):
    """Decorator used to specify a CPU function used to verify the output of
    a GPU function."""

    def debug_func(gpu_func):
        @wraps(gpu_func)
        def wrapped_func(*args, **kwargs):
            if debug:
                cpu_args = list(args)
                for i, a in enumerate(cpu_args):
                    if isinstance(a, gpuarray.GPUArray):
                        cpu_args[i] = a.get()
                cpu_kwargs = copy.copy(kwargs)
                for k in cpu_kwargs:
                    if isinstance(cpu_kwargs[k], gpuarray.GPUArray):
                        cpu_kwargs[k] = cpu_kwargs[k].get()

                out_cpu = cpu_func(*cpu_args, **cpu_kwargs)

            out_gpu = gpu_func(*args, **kwargs)

            if debug:
                try:
                    tmp = out_gpu.get()
                    assert np.allclose(tmp.ravel(), out_cpu.ravel(), rtol=1e-4)
                except AssertionError:
                    print np.max(np.abs(out_cpu - tmp))
                    print "gpu"
                    print tmp
                    print "cpu"
                    print out_cpu
                    print "gpu/cpu"
                    print tmp / out_cpu
                    raise
            return out_gpu
        return wrapped_func
    return debug_func


def cpu_dot(a, b, out=None, transpose_a=False, transpose_b=False,
            increment=False):
    result = np.dot(a.T if transpose_a else a, b.T if transpose_b else b)

    return result + out * increment


def cpu_sum_cols(a, out=None, increment=False):
    result = np.sum(a, axis=0)

    return result + out * increment


def cpu_J_dot(J, v, transpose_J=False, out=None, increment=False):
    if J.ndim == 2:
        result = J * v
    else:
        if transpose_J:
            J = np.transpose(J, (0, 2, 1))
        result = np.einsum("ijk,ik->ij", J, v)

    return result + out * increment


def cpu_multiply(a, b, out=None, increment=False):
    return a * b + out * increment


# @debug_wrapper(cpu_dot, debug=True)
def dot(a, b, out=None, transpose_a=False, transpose_b=False,
        increment=False):

    assert not increment or out is not None

    if transpose_a:
        a1, a0 = a.shape
    else:
        a0, a1 = a.shape

    if transpose_b:
        b1, b0 = b.shape
    else:
        b0, b1 = b.shape

    assert a1 == b0

    if out is None:
        out = gpuarray.zeros((a0, b1), dtype=np.float32)

    # note: we swap the order of a and b and swap the transposes because
    # cublas assumes column-major ordering
    transa = "t" if transpose_a else "n"
    transb = "t" if transpose_b else "n"
    beta = np.float32(1.0) if increment else np.float32(0.0)
    lda = a0 if transpose_a else a1
    ldb = b0 if transpose_b else b1
    ldout = b1

    cublas.cublasSgemm(misc._global_cublas_handle, transb, transa, b1, a0, a1,
                       np.float32(1.0), b.gpudata, ldb, a.gpudata, lda,
                       beta, out.gpudata, ldout)

    return out


# @debug_wrapper(cpu_J_dot, True)
def J_dot(J, v, out=None, transpose_J=False, increment=False):
    if J.ndim == 2:
        return multiply(J, v, out=out, increment=increment)

    if out is v:
        # note: we allow out to be v in this function because it can be
        # handled efficiently in the element-wise case
        v = v.copy()

    if transpose_J:
        a1, a0 = J.shape[1:]
    else:
        a0, a1 = J.shape[1:]

    # note: all the transposes are swapped because cublas assumes column-major
    # ordering
    lda = a0 if transpose_J else a1
    beta = np.float32(1.0) if increment else np.float32(0.0)

    if out is None:
        out = gpuarray.zeros((J.shape[0], a1), dtype=np.float32)

    # concurrent gemv approach (this seems to be slower than batched)
#     transJ = "n" if transpose_J else "t"
#     streams = [driver.Stream() for _ in range(J.shape[0])]
#     for i in range(J.shape[0]):
#         cublas.cublasSetStream(misc._global_cublas_handle, streams[i].handle)
#         cublas.cublasSgemv(misc._global_cublas_handle, transJ, a1, a0,
#                            np.float32(1.0), J[i].gpudata, lda, v[i].gpudata, 1,
#                            beta, out[i].gpudata, 1)
#
#     cublas.cublasSetStream(misc._global_cublas_handle, 0)

    # batched gemm approach
    transJ = "t" if transpose_J else "n"
    J_data = gpuarray.arange(np.int64(J.gpudata),
                             np.int64(J.gpudata) + 4 * a0 * a1 * J.shape[0],
                             4 * a0 * a1, dtype=np.int64)
    v_data = gpuarray.arange(np.int64(v.gpudata),
                             np.int64(v.gpudata) + 4 * a1 * v.shape[0],
                             4 * a1, dtype=np.int64)
    out_data = gpuarray.arange(np.int64(out.gpudata),
                               np.int64(out.gpudata) + 4 * a0 * out.shape[0],
                               4 * a0, dtype=np.int64)

    cublas.cublasSgemmBatched(
        misc._global_cublas_handle, "n", transJ, 1, a0, a1,
        np.float32(1.0), v_data.gpudata, 1, J_data.gpudata, lda,
        beta, out_data.gpudata, 1, J.shape[0])

    return out


# @debug_wrapper(cpu_sum_cols, debug=True)
def sum_cols(a, out=None, increment=False):
    if out is None:
        out = gpuarray.empty(int(a.shape[1]), dtype=np.float32)

    block_x = min(32, a.shape[1])
    block_y = 32

    grid = (a.shape[1] / block_x + (a.shape[1] % block_x != 0), 1)

    gpu_sum_cols = hf.gpu.kernels.get_function("sum_cols")
    gpu_sum_cols(a, out, np.int32(increment),
                 np.int32(a.shape[0]), np.int32(a.shape[1]),
                 block=(block_x, block_y, 1), grid=grid,
                 shared=block_x * block_y * 4)

    return out


# @debug_wrapper(lambda a, b: a + b, True)
def iadd(a, b):
    block_x = min(32, a.shape[1])
    block_y = min(32, a.shape[0])
    grid = (a.shape[1] / block_x + (a.shape[1] % block_x != 0),
            a.shape[0] / block_y + (a.shape[0] % block_y != 0))

    gpu_add = hf.gpu.kernels.get_function("iadd")
    gpu_add(a, b, np.int32(a.shape[0]), np.int32(a.shape[1]),
            block=(block_x, block_y, 1), grid=grid)

    return a


# @debug_wrapper(cpu_multiply, True)
def multiply(a, b, out=None, increment=False):
    assert a.size == b.size

    if out is None:
        out = gpuarray.zeros(a.shape, dtype=np.float32)

    block_size = min(128, a.size)
    gpu_multiply = hf.gpu.kernels.get_function("multiply")
    gpu_multiply(a, b, out, np.int32(a.size), np.int32(increment),
                 block=(block_size, 1, 1),
                 grid=(a.size / block_size + (a.size % block_size != 0), 1))

    return out
