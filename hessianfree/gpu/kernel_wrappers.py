import copy
from functools import wraps

import numpy as np
from pycuda import gpuarray, driver
from skcuda import cublas, misc

import hessianfree as hf

misc.init()


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

    # note: all the transposes are swapped because cublas assumes column-major
    # ordering
    lda = a0 if transpose_J else a1
    beta = dtype.type(1.0) if increment else dtype.type(0.0)

    if out is v:
        # note: we allow out to be v in this function because it can be
        # handled efficiently in the element-wise case
        v = v.copy()
    elif out is None:
        out = gpuarray.zeros((J.shape[0], a1), dtype=dtype)

    assert J.dtype == v.dtype == out.dtype

    # concurrent gemv approach (this seems to be slower than batched)
#     transJ = "n" if transpose_J else "t"
#     for i in range(J.shape[0]):
#         cublas.cublasSetStream(misc._global_cublas_handle,
#                                multi_streams[i % len(multi_streams)].handle)
#         cublas.cublasSgemv(misc._global_cublas_handle, transJ, a1, a0,
#                            np.float32(1.0), J[i].gpudata, lda, v[i].gpudata, 1,
#                            beta, out[i].gpudata, 1)
#
#     cublas.cublasSetStream(misc._global_cublas_handle, 0)

    # batched gemm approach
    transJ = "t" if transpose_J else "n"
    n = np.dtype(dtype).itemsize
    J_data = gpuarray.arange(np.int64(J.gpudata),
                             np.int64(J.gpudata) + n * a0 * a1 * J.shape[0],
                             n * a0 * a1, dtype=np.int64, stream=stream)
    v_data = gpuarray.arange(np.int64(v.gpudata),
                             np.int64(v.gpudata) + n * a1 * v.shape[0],
                             n * a1, dtype=np.int64, stream=stream)
    out_data = gpuarray.arange(np.int64(out.gpudata),
                               np.int64(out.gpudata) + n * a0 * out.shape[0],
                               n * a0, dtype=np.int64, stream=stream)

    if stream is None:
        cublas.cublasSetStream(misc._global_cublas_handle, 0)
    else:
        cublas.cublasSetStream(misc._global_cublas_handle, stream.handle)

    if dtype == np.float32:
        gemmBatched = cublas.cublasSgemmBatched
    else:
        gemmBatched = cublas.cublasDgemmBatched

    gemmBatched(misc._global_cublas_handle, "n", transJ, 1, a0, a1,
                dtype.type(1.0), v_data.gpudata, 1, J_data.gpudata, lda,
                beta, out_data.gpudata, 1, J.shape[0])

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

    block_size = min(128, a.size)
    hf.gpu.multiply_kernel[dtype == np.float32].prepared_async_call(
        (a.size / block_size + (a.size % block_size != 0), 1),
        (block_size, 1, 1), stream,
        a.gpudata, b.gpudata, out.gpudata, np.int32(a.size),
        np.int32(increment))

    return out


def shared_dot(a, b, out=None, transpose_a=False, transpose_b=False,
               increment=False, stream=None, shortcut=True):
    # non-cublas matrix multiplication

    if (shortcut and a.shape[transpose_a] < 512 and
            b.shape[1 - transpose_b] < 16):
        return mv_dot(a, b, out=out, transpose_a=transpose_a,
                      transpose_v=transpose_b, batch_a=False, batch_v=True,
                      increment=increment)
    elif (shortcut and b.shape[1 - transpose_b] < 512 and
            a.shape[transpose_a] < 16):
        return mv_dot(b, a, out=out, transpose_a=not transpose_b,
                      transpose_v=not transpose_a, batch_a=False, batch_v=True,
                      increment=increment, transpose_out=True)

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


def mv_dot(a, v, out=None, transpose_a=False, transpose_v=False,
           transpose_out=False, batch_a=False, batch_v=False, increment=False,
           stream=None):

    dtype = a.dtype

    # transpose_v only applies if v is batched
    assert not transpose_v or batch_v

    # if a and v are both batched, then the batches must align
    assert (not (batch_a and batch_v) or
            (transpose_v and a.shape[0] == v.shape[0]))

    assert out is None or (out is not a and out is not v)

    if batch_a:
        grid_y = a.shape[0]
    elif batch_v:
        grid_y = v.shape[1 - transpose_v]
    else:
        grid_y = 1

    a_shape = (a.shape[0 + batch_a], a.shape[1 + batch_a])
    if transpose_a:
        a_shape = (a_shape[1], a_shape[0])

    if out is None:
        if batch_a != transpose_out:
            out_shape = (grid_y, a_shape[0])
        else:
            out_shape = (a_shape[0], grid_y)
        out = gpuarray.empty(out_shape, dtype=dtype)

    assert a.dtype == v.dtype == out.dtype

    block_x = block_y = 32
    grid = (a_shape[0] / block_x + (a_shape[0] % block_x != 0), grid_y)

    hf.gpu.mv_dot_kernel[dtype == np.float32
                         ][transpose_a][transpose_v].prepared_async_call(
        grid, (block_x, block_y, 1), stream,
        a.gpudata, v.gpudata, out.gpudata, np.int32(batch_a),
        np.int32(batch_v), np.int32(a_shape[0]), np.int32(a_shape[1]),
        np.int32(increment), np.int32(transpose_out))

    return out
