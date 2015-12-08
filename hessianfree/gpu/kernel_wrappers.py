import copy
from functools import wraps
import warnings

import numpy as np
from pycuda import gpuarray
from skcuda import cublas, misc

import hessianfree as hf

misc.init()


# TODO: pick block sizes more carefully


def debug_wrapper(cpu_func, debug=False):
    """Decorator used to specify a CPU function used to verify the output of
    a GPU function."""

    def debug_func_parametrized(gpu_func):
        @wraps(gpu_func)
        def debug_func(*args, **kwargs):
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
        return debug_func
    return debug_func_parametrized


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
def cublas_dot(a, b, out=None, transpose_a=False, transpose_b=False,
               increment=False, stream=None):

    assert not increment or out is not None

    dtype = a.dtype

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
        out = gpuarray.zeros((a0, b1), dtype=dtype)

    assert a.dtype == b.dtype == out.dtype

    # note: we swap the order of a and b and swap the transposes because
    # cublas assumes column-major ordering
    transa = "t" if transpose_a else "n"
    transb = "t" if transpose_b else "n"
    beta = dtype.type(1.0) if increment else dtype.type(0.0)
    lda = a0 if transpose_a else a1
    ldb = b0 if transpose_b else b1
    ldout = b1

    if stream is None:
        # TODO: how much overhead is there in this? should we try to minimize
        # the number of calls by checking if stream is already set?
        cublas.cublasSetStream(misc._global_cublas_handle, 0)
    else:
        cublas.cublasSetStream(misc._global_cublas_handle, stream.handle)

    if dtype == np.float32:
        gemm = cublas.cublasSgemm
    else:
        gemm = cublas.cublasDgemm

    gemm(misc._global_cublas_handle, transb, transa, b1, a0, a1,
         dtype.type(1.0), b.gpudata, ldb, a.gpudata, lda, beta, out.gpudata,
         ldout)

    return out


# @debug_wrapper(cpu_J_dot, True)
def J_dot(J, v, out=None, transpose_J=False, increment=False, stream=None):
    if J.ndim == 2:
        return multiply(J, v, out=out, increment=increment, stream=stream)

    dtype = J.dtype

    if transpose_J:
        a1, a0 = J.shape[1:]
    else:
        a0, a1 = J.shape[1:]

    if out is None:
        out = gpuarray.zeros((J.shape[0], a1), dtype=dtype)
    elif out is v:
        # note: we allow out to be v in this function because it can be
        # handled efficiently in the element-wise case
        warnings.warn("Copying v in J_dot")
        v = v.copy()

    assert J.dtype == v.dtype == out.dtype

    # mv kernel approach
    block = (32, 32, 1)
    grid = (a0 / 32 + (a0 % 32 != 0), J.shape[0])
    hf.gpu.mv_batched_kernel[dtype ==
                             np.float32][transpose_J].prepared_async_call(
        grid, block, stream, J.gpudata, v.gpudata, out.gpudata,
        np.int32(a0), np.int32(a1), np.int32(increment),
        shared_size=1088 * dtype.itemsize)

    # concurrent gemv approach (this seems to be slower, but maybe for
    # really large matrices?)
    # note: all the transposes are swapped because cublas assumes column-major
    # ordering
#     lda = a0 if transpose_J else a1
#     beta = dtype.type(1.0) if increment else dtype.type(0.0)
#     transJ = "n" if transpose_J else "t"
#     if dtype == np.float32:
#         gemv = cublas.cublasSgemv
#     else:
#         gemv = cublas.cublasDgemv
#     for i in range(J.shape[0]):
#         cublas.cublasSetStream(misc._global_cublas_handle,
#                                hf.gpu.streams[i % len(hf.gpu.streams)].handle)
#         gemv(misc._global_cublas_handle, transJ, a1, a0, np.float32(1.0),
#              J[i].gpudata, lda, v[i].gpudata, 1, beta, out[i].gpudata, 1)
#
#     cublas.cublasSetStream(misc._global_cublas_handle, 0)

    return out


# @debug_wrapper(cpu_sum_cols, debug=True)
def sum_cols(a, out=None, increment=False, stream=None):
    dtype = a.dtype

    if out is None:
        out = gpuarray.empty(int(a.shape[1]), dtype=dtype)

    assert a.dtype == out.dtype

    block_x = min(32, a.shape[1])
    block_y = 32

    grid = (a.shape[1] / block_x + (a.shape[1] % block_x != 0), 1)
    hf.gpu.sum_cols_kernel[dtype == np.float32].prepared_async_call(
        grid, (block_x, block_y, 1), stream,
        a.gpudata, out.gpudata, np.int32(increment), np.int32(a.shape[0]),
        np.int32(a.shape[1]),
        shared_size=block_x * block_y * dtype.itemsize)

    return out


# @debug_wrapper(lambda a, b: a + b, True)
def iadd(a, b, stream=None):
    dtype = a.dtype
    assert a.dtype == b.dtype

    block_x = min(32, a.shape[1])
    block_y = min(32, a.shape[0])
    grid = (a.shape[1] / block_x + (a.shape[1] % block_x != 0),
            a.shape[0] / block_y + (a.shape[0] % block_y != 0))

    hf.gpu.iadd_kernel[dtype == np.float32].prepared_async_call(
        grid, (block_x, block_y, 1), stream,
        a.gpudata, b.gpudata, np.int32(a.shape[0]), np.int32(a.shape[1]))

    return a


# @debug_wrapper(cpu_multiply, True)
def multiply(a, b, out=None, increment=False, stream=None):
    dtype = a.dtype

    if out is None:
        out = gpuarray.zeros(a.shape, dtype=dtype)

    assert a.size == b.size
    assert a.dtype == b.dtype == out.dtype

    hf.gpu.multiply_kernel[dtype == np.float32].prepared_async_call(
        a._grid, a._block, stream,
        a.gpudata, b.gpudata, out.gpudata, np.int32(a.size),
        np.int32(increment))

    return out


def shared_dot(a, b, out=None, transpose_a=False, transpose_b=False,
               increment=False, stream=None):
    # non-cublas matrix multiplication

    dtype = a.dtype

    # note: these transposes don't actually rearrange anything in memory,
    # just changing the shape
    if transpose_a:
        a = a.T
    if transpose_b:
        b = b.T

    assert a.shape[1] == b.shape[0]
    assert out is None or (out is not a and out is not b)

    if out is None:
        out = gpuarray.zeros((a.shape[0], b.shape[1]), dtype=dtype)

    assert a.dtype == b.dtype == out.dtype

    # note: the block is transposed from what you might think, so it's
    # (b_shape[1], a_shape[0]), because we want the x threads aligned with
    # rows to support memory coalescing
    block_x = block_y = 32
    grid = (b.shape[1] / block_x + (b.shape[1] % block_x != 0),
            a.shape[0] / block_y + (a.shape[0] % block_y != 0))

    hf.gpu.m_dot_kernel[dtype == np.float32
                        ][transpose_a][transpose_b].prepared_async_call(
        grid, (block_x, block_y, 1), stream,
        a.gpudata, b.gpudata, out.gpudata, np.int32(a.shape[0]),
        np.int32(a.shape[1]), np.int32(b.shape[1]), np.int32(increment),
        shared_size=(block_x + 1) * block_y * 2 * dtype.itemsize)

    return out
