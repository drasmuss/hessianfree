import copy

import numpy as np
from pycuda import gpuarray, driver
from pycuda.autoinit import device, context

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


# def gpu_wrapper(gpu_func):
#     @wraps(gpu_func)
#     def wrapped_gpu(*args, **kwargs):
#         for i, a in enumerate(args):
#             if isinstance(a, np.ndarray):
#                 args[i] = gpuarray.to_gpu(a)
#
#         out_cpu = None
#         if (kwargs.get("out", None) is not None and
#                 isinstance(kwargs["out"], np.ndarray)):
#             out_cpu = kwargs["out"]
#             kwargs["out"] = gpuarray.to_gpu(out_cpu)
#
#         out = gpu_func(*args, **kwargs)
#         if out_cpu is not None:
#             out.get(out_cpu)
#             out = out_cpu
#
#         return out
#
#     return wrapped_gpu


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
                    assert np.allclose(tmp, out_cpu, rtol=1e-5)
                except AssertionError:
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
    else:
        return result


@debug_wrapper(cpu_m_dot, debug=False)
def m_dot(a, b, out=None, transpose_a=False, transpose_b=False,
          increment=False):

    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))
    b_shape = (np.int32(b.shape[0]), np.int32(b.shape[1]))
    if transpose_a:
        a_shape = (a_shape[1], a_shape[0])
    if transpose_b:
        b_shape = (b_shape[1], b_shape[0])

    assert a_shape[1] == b_shape[0]
    assert not increment or out is not None

    if out is None:
        out = gpuarray.zeros((a_shape[0], b_shape[1]), dtype=np.float32)

    assert type(a) == type(b) == type(out) == gpuarray.GPUArray

    block_x = block_y = np.int32(32)
    grid = (b_shape[1] / block_y + (b_shape[1] % block_y != 0),
            a_shape[0] / block_x + (a_shape[0] % block_x != 0))

#     print "multiplying", a.shape, b.shape, out.shape
#     print type(a), type(b), type(out)
#     print a.dtype, b.dtype, out.dtype
#     print transpose_a, transpose_b
#     print a_shape[0], a_shape[1], b_shape[1], tile_len
#     print "block", block_x, block_y
#     print "grid", grid
#     print a
#     print b

    # note: the thread is transposed from what you might think, so it's
    # (b_shape[1], a_shape[0]), because we want the x threads aligned with
    # rows, to support memory coalescing

    context.synchronize()  # TODO: why does this speed things up?
    hf.gpu.m_dot_kernel[transpose_a][transpose_b](
        a, b, out, a_shape[0], a_shape[1], b_shape[1], np.int32(increment),
        grid=grid, block=(block_y, block_x, 1))

    return out


def J_dot(a, b, out=None, transpose_a=False):
    if a.ndim == 2:
        # element-wise case
        if out is None:
            return a * b
        elif out is b:
            out *= a
            return out
        else:
            out[...] = b
            out *= a
            return out

    # note: this is basically m_dot, except we know that b is a vector
    # (which we'll just treat as an nx1 matrix), and we use asynchronous
    # streams to compute the different batches
    # TODO: implement a dedicated J_dot, would be more efficient (but only
    # for non-diagonal J, so not a high priority)

    # note: this is not totally thread safe if out=b. we could have problems if
    # one block finishes and writes to out before another block has loaded that
    # part of b into shared memory. however, in practice it seems that blocks
    # never get that out of sync.

    # we're going to swap a and b because m_dot is designed to align warps
    # along b1, but here b1 is always 1, which will
    # not be efficient.  so instead of computing C = A*B we'll compute
    # C=(B^T*A^T)^T.
    tmp = a
    a = b
    b = tmp
    transpose_b = not transpose_a
    transpose_a = False

    n_batches = a.shape[0]
    a_shape = (np.int32(1), np.int32(a.shape[1]))
    b_shape = (np.int32(b.shape[1]), np.int32(b.shape[2]))
    if transpose_b:
        b_shape = (b_shape[1], b_shape[0])

    assert a_shape[1] == b_shape[0]

    if out is None:
        out = gpuarray.zeros((n_batches, b_shape[1]), dtype=np.float32)

    assert type(a) == type(b) == type(out) == gpuarray.GPUArray

    block_x = block_y = np.int32(32)
    grid = (b_shape[1] / block_y + (b_shape[1] % block_y != 0), 1)
    # TODO: parallelize things across the grid rather than across streams?

    context.synchronize()  # TODO: why does this speed things up?
    for i in range(n_batches):
        hf.gpu.m_dot_kernel[transpose_a][transpose_b](
            a[i], b[i], out[i], a_shape[0], a_shape[1], b_shape[1],
            np.int32(False), stream=driver.Stream(),
            grid=grid, block=(block_y, block_x, 1))

    # this synchronize is only really necessary if J_dot were being executed
    # asynchronously itself, which is never the case in the current
    # implementation. but there's going to be an implicit synchronize anyway
    # since it's running in the default stream, so doesn't hurt
    context.synchronize()

    return out


@debug_wrapper(cpu_m_dot, debug=False)
def simple_m_dot(a, b, out=None, transpose_b=False, increment=False):
    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))
    b_shape = (np.int32(b.shape[0]), np.int32(b.shape[1]))
    if transpose_b:
        b_shape = (b_shape[1], b_shape[0])

    assert a_shape[1] == b_shape[0]

    if out is None:
        out = gpuarray.zeros((a_shape[0], b_shape[1]), dtype=np.float32)

    assert type(a) == type(b) == type(out) == gpuarray.GPUArray

    block_y = find_block_len(device.MAX_THREADS_PER_BLOCK, 4, b_shape[1])
    block_x = find_block_len(device.MAX_THREADS_PER_BLOCK, block_y, a_shape[0])

#     print "multiplying", a.shape, b.shape, out.shape, a_shape[1], np.int32(transpose_b), np.int32(increment)
#     print a.dtype, b.dtype, out.dtype, type(a), type(b), type(out)
#     print block_x, block_y
#     print (a_shape[0] / block_x, b_shape[1] / block_y)
#     print pycuda.driver.mem_get_info()
#     print a
#     print b

    gpu_m_dot = hf.gpu.kernels.get_function("simple_m_dot")
    gpu_m_dot(a, b, out, a_shape[1],
              np.int32(transpose_b), np.int32(increment),
              grid=(a_shape[0] / block_x, b_shape[1] / block_y),
              block=(block_x, block_y, 1))

    return out


@debug_wrapper(lambda a, axis, **_: np.sum(a, axis=axis), False)
def sum_axis(a, axis, out=None, increment=False):
    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))

    if out is None:
        out = gpuarray.zeros(a_shape[1 - axis], dtype=np.float32)

#     block_len = find_block_len(device.MAX_THREADS_PER_BLOCK, 1,
#                                a_shape[1 - axis])
    block_len = np.int32(min(32, a.shape[1 - axis]))

    gpu_sum_axis = hf.gpu.kernels.get_function("sum_axis")
    gpu_sum_axis(a, out, np.int32(axis), a_shape[0], a_shape[1],
                 np.int32(increment),
                 grid=(a_shape[1 - axis] / block_len +
                       (a_shape[1 - axis] % block_len != 0), 1),
                 block=(block_len, 1, 1))

    return out


@debug_wrapper(lambda a, b: a + b, False)
def iadd(a, b):
    block_x = np.int32(min(32, a.shape[1]))
    block_y = np.int32(min(32, a.shape[0]))
    grid = (a.shape[1] / block_x + (a.shape[1] % block_x != 0),
            a.shape[0] / block_y + (a.shape[0] % block_y != 0))

    gpu_add = hf.gpu.kernels.get_function("iadd")
    gpu_add(a, b, np.int32(a.shape[0]), np.int32(a.shape[1]),
            block=(block_x, block_y, 1), grid=grid)

    return a
