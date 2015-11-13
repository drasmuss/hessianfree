import copy

import numpy as np
from pycuda import gpuarray

import hessianfree as hf
from functools import wraps


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
                    assert np.allclose(tmp.ravel(), out_cpu.ravel(), atol=1e-5)
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


def cpu_m_dot(a, b, out=None, transpose_a=False, transpose_b=False,
              increment=False):
    # equivalent function implementation on the cpu, for debugging
    result = np.dot(a.T if transpose_a else a, b.T if transpose_b else b)
    if increment:
        return result + out
    return result


def cpu_sum_cols(a, out=None, increment=False):
    result = np.sum(a, axis=0)
    if increment:
        return result + out
    return result


def cpu_J_dot(J, v, transpose_J=False, out=None, increment=False):
        if J.ndim == 2:
            result = J * v
        else:
            if transpose_J:
                J = np.transpose(J, (0, 2, 1))
            result = np.einsum("ijk,ik->ij", J, v)

        if increment:
            return result + out
        return result


# @debug_wrapper(cpu_m_dot, debug=True)
def m_dot(a, b, out=None, transpose_a=False, transpose_b=False,
          increment=False, shortcut=False):

    if (shortcut and a.shape[transpose_a] < 512 and
            b.shape[1 - transpose_b] < 16):
        # TODO: do some transpose thing so we can shortcut small a
        return mv_dot(a, b, out=out, transpose_a=transpose_a,
                      transpose_v=transpose_b, batch_a=False, batch_v=True,
                      increment=increment)

    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))
    b_shape = (np.int32(b.shape[0]), np.int32(b.shape[1]))
    if transpose_a:
        a_shape = (a_shape[1], a_shape[0])
    if transpose_b:
        b_shape = (b_shape[1], b_shape[0])

    assert a_shape[1] == b_shape[0]

    if out is None:
        out = gpuarray.zeros((a_shape[0], b_shape[1]), dtype=np.float32)

    block_x = block_y = np.int32(32)
    grid = (b_shape[1] / block_x + (b_shape[1] % block_x != 0),
            a_shape[0] / block_y + (a_shape[0] % block_y != 0))

    # note: the block is transposed from what you might think, so it's
    # (b_shape[1], a_shape[0]), because we want the x threads aligned with
    # rows to support memory coalescing
    hf.gpu.m_dot_kernel[transpose_a][transpose_b](
        a, b, out, a_shape[0], a_shape[1], b_shape[1], np.int32(increment),
        grid=grid, block=(block_x, block_y, 1),
        shared=(block_x + 1) * block_y * 8)

    return out


# @debug_wrapper(cpu_J_dot, True)
def J_dot(J, v, out=None, transpose_J=False, increment=False):
    if J.ndim == 2:
        # element-wise case
        if out is None:
            out = J * v
        elif increment:
            out += J * v
        elif out is v:
            out *= J
        else:
            out[...] = v
            out *= J
        return out

    if out is v:
        tmp_v = v.copy()
    else:
        tmp_v = v

    return mv_dot(J, tmp_v, out=out, transpose_a=transpose_J, transpose_v=True,
                  batch_a=True, batch_v=True, increment=increment)


# @debug_wrapper(cpu_sum_cols, debug=True)
def sum_cols(a, out=None, increment=False):
    if out is None:
        out = gpuarray.empty(int(a.shape[1]), dtype=np.float32)

    block_x = min(32, a.shape[1])
    block_y = 32

    # TODO: is it faster to make block_y evenly divide a[0]?

    grid = (a.shape[1] / block_x + (a.shape[1] % block_x != 0),
            a.shape[0] / block_y + (a.shape[0] % block_y != 0))

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


def mv_dot(a, v, out=None, transpose_a=False, transpose_v=False,
           batch_a=False, batch_v=False, increment=False):

    if batch_a:
        grid_y = a.shape[0]
    elif batch_v:
        grid_y = v.shape[1 - transpose_v]
    else:
        grid_y = 1

    # transpose_v only applies if there is a degree of freedom there
    assert not transpose_v or batch_v

    # if a and v are both batched, then the batches must align
    assert (not (batch_a and batch_v) or
            (transpose_v and a.shape[0] == v.shape[0]))

    a_shape = (np.int32(a.shape[0 + batch_a]), np.int32(a.shape[1 + batch_a]))
    if transpose_a:
        a_shape = (a_shape[1], a_shape[0])

    if out is None:
        if batch_a:
            out_shape = (grid_y, a_shape[0])
        else:
            out_shape = (a_shape[0], grid_y)
        out = gpuarray.empty(out_shape, dtype=np.float32)

    block_x = block_y = 32
    grid = (a_shape[0] / block_x + (a_shape[0] % block_x != 0), grid_y)

    hf.gpu.mv_dot_kernel[transpose_a][transpose_v](
        a, v, out, np.int32(batch_a), np.int32(batch_v),
        a_shape[0], a_shape[1], np.int32(increment),
        block=(block_x, block_y, 1), grid=grid)

    return out
