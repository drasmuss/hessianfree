import numpy as np
from pycuda import gpuarray
from pycuda.autoinit import device, context

import hessianfree as hf


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


def outer_sum(a, b, out=None):
    a_len = np.int32(a.shape[1])
    b_len = np.int32(b.shape[1])
    batchsize = np.int32(a.shape[0])  # assume == b.shape[0]

    if out is None:
        out = gpuarray.zeros(a_len * b_len, np.float32)

    assert a.dtype == b.dtype == np.float32

    cols_per_block = find_block_len(device.MAX_THREADS_PER_BLOCK, 1, b_len)
    rows_per_block = find_block_len(device.MAX_THREADS_PER_BLOCK,
                                    cols_per_block, a_len)

    # execute function
    gpu_outer = hf.gpu.mod.get_function("outer_sum")
    gpu_outer(a, b, out, batchsize,
              grid=(a_len / rows_per_block, b_len / cols_per_block),
              block=(rows_per_block, cols_per_block, 1))

    return out


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

    # TODO: customize block size to GPU compute capability?
    block_x = block_y = np.int32(32)
    tile_len = block_x
#    tile_len = min(np.int32(device.MAX_SHARED_MEMORY_PER_BLOCK / 8 / block_x),
#                   (a_shape[1] / block_x +
#                    (a_shape[1] % block_x != 0)) * block_x)
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

#     assert block_x * tile_len < device.MAX_SHARED_MEMORY_PER_BLOCK / 8
#     assert block_y * tile_len < device.MAX_SHARED_MEMORY_PER_BLOCK / 8

    # note: the thread is transposed from what you might think
    # (b_shape[1], a_shape[0]), because we want the x threads aligned with
    # rows, to support memory coalescing

    context.synchronize()  # TODO: why does this speed things up?
    hf.gpu.m_dot_kernel[transpose_a][transpose_b](
        a, b, out, a_shape[0], a_shape[1], b_shape[1], np.int32(increment),
        grid=grid, block=(block_y, block_x, 1))

#     print "old block", block_x, block_y
#
#     block_y = find_block_len(device.MAX_THREADS_PER_BLOCK, 4, b_shape[1])
#     block_x = find_block_len(device.MAX_THREADS_PER_BLOCK, block_y, a_shape[0])
#
#     print "new block", block_x, block_y
#
#     gpu_m_dot = mod.get_function("simple_m_dot")
#     gpu_m_dot(a, b, out, a_shape[1],
#               np.int32(transpose_b), np.int32(increment),
#               grid=(a_shape[0] / block_x, b_shape[1] / block_y),
#               block=(block_x, block_y, 1))

    return out


def simple_m_dot(a, b, out=None, transpose_b=False, increment=False):
    # TODO: we could just load the rows into shared memory, since they tend
    # to be long

    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))
    b_shape = (np.int32(b.shape[0]), np.int32(b.shape[1]))
    if transpose_b:
        b_shape = (b_shape[1], b_shape[0])

    assert a_shape[1] == b_shape[0]
    assert not increment or out is not None

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

    gpu_m_dot = hf.gpu.mod.get_function("simple_m_dot")
    gpu_m_dot(a, b, out, a_shape[1],
              np.int32(transpose_b), np.int32(increment),
              grid=(a_shape[0] / block_x, b_shape[1] / block_y),
              block=(block_x, block_y, 1))

    return out


def sum_axis(a, axis, out=None):
    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))

    if out is None:
        out = gpuarray.zeros(a_shape[1 - axis], dtype=np.float32)

    block_len = find_block_len(device.MAX_THREADS_PER_BLOCK, 1,
                               a_shape[1 - axis])

    gpu_sum_axis = hf.gpu.mod.get_function("sum_axis")
    gpu_sum_axis(a, out, np.int32(axis), a_shape[0], a_shape[1],
                 grid=(a_shape[1 - axis] / block_len, 1),
                 block=(block_len, 1, 1))

    return out
