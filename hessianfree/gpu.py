import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pycuda.autoinit import device

import numpy as np

mod = SourceModule("""
#include <stdio.h>

__global__ void outer_sum(float *a, float *b, float *out,
                          int batch_size)
{
    int a_i = blockIdx.x*blockDim.x + threadIdx.x;
    int b_i = blockIdx.y*blockDim.y + threadIdx.y;
    const int a_len = blockDim.x * gridDim.x;
    const int b_len = blockDim.y * gridDim.y;
    const int out_addr = a_i*b_len + b_i;

    out[out_addr] = 0;
    for(int j=0; j < batch_size; j++) {
        out[out_addr] += a[a_i] * b[b_i];
        a_i += a_len;
        b_i += b_len;
    }

    // TODO: convert this to a tiled approach like m_dot
}

__global__ void m_dot(float *A, float *B, float *C, int a1,
                      int tile_len, int transpose_b, int increment)
{
    // multiplying an [a0,a1] matrix by an [a1,b1] matrix. each thread will
    // compute one element of c, which is a[i,:] * b[:,j].  however, we can
    // make this more efficient by loading a[i,:] into shared memory, and
    // then computing all the c[i,:] elements off that. we probably can't fit
    // all the necessary data into shared memory, so we load in a[i,:tile_len]
    // and b[:tile_len, j] sized chunks one at a time.

    // each c[i,j] thread will be responsible for loading in a[i,:n] and
    // b[:m, j] into shared memory, where n/m will depend on the relationship
    // between tile_len and the thread block dimensions

    // note that if the data is small enough to all fit in shared memory then
    // we don't need to worry about the tiling

    // thread variables
    const int b_x = blockDim.x;
    const int b_y = blockDim.y;
    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;

    const int b1 = b_y*gridDim.y;

    // row/col for this thread
    const int a_i = blockIdx.x*b_x + t_x;
    const int b_j = blockIdx.y*b_y + t_y;

    // how many entries is this thread responsible for loading
    int base_n = tile_len / b_y;
    int extra_n = tile_len %% b_y;
    int n = base_n;
    int n_off = base_n*t_y;
    if (t_y < extra_n)
    {
        n += 1;
        n_off += t_y;
    }
    else
        n_off += extra_n;

    int base_m = tile_len / b_x;
    int extra_m = tile_len %% b_x;
    int m = base_m;
    int m_off = base_m*t_x;
    if (t_x < extra_m)
    {
        m += 1;
        m_off += t_x;
    }
    else
        m_off += extra_m;

    // offsets for the matrix areas this thread is responsible for
    const int A_tile_off = t_x*tile_len + n_off;
    const int A_off = a_i*a1 + n_off;
    const int B_tile_off = m_off*b_y + t_y;
    // const int B_off = m_off*

    // printf("thread (%%d, %%d)\\n n %%d n_off %%d\\n m %%d m_off %%d\\n",
    //        a_i, b_j, n, n_off, m, m_off);

    // c will accumulate the value of c[i,j] across tiles
    float c = 0;
    if (increment)
    {
        c = C[a_i*b1 + b_j];
    }

    // loop over the tiles
    for (int tile_i=0; tile_i < a1; tile_i += tile_len)
    {
        // create the tile
        // TODO: allocate the actual necessary size
        __shared__ float A_tile[%(SHARED_MEM)s / 8]; //[b_x][tile_len];
        __shared__ float B_tile[%(SHARED_MEM)s / 8]; //[tile_len][b_y];

        // each thread loads in its part of A/B
        for (int i=0; i < n; i++)
            A_tile[A_tile_off+i] = A[A_off + tile_i + i];

        if (!transpose_b)
        {
            for (int i=0; i < m; i++)
                B_tile[B_tile_off + i*b_y] = B[(tile_i+m_off+i)*b1 + b_j];
        }
        else
        {
            for (int i=0; i < m; i++)
                B_tile[B_tile_off + i*b_y] = B[b_j*a1 + tile_i+m_off+i];
        }

        // wait for all threads to finish loading in their data
        __syncthreads();

        /*
        if (a_i == 0 && b_j == 0)
        {
            for(int i=0; i < b_x * tile_len; i++)
                printf("%%f ", A_tile[i]);
            printf("\\n");
            for(int i=0; i < b_y * tile_len; i++)
                printf("%%f ", B_tile[i]);
            printf("\\n");
            for(int i=0; i < b_y * tile_len; i++)
                printf("%%f ", B[i]);
            printf("\\n");
        }
        */

        // accumulate the product for this thread
        for (int i=0; i < tile_len; i++)
            c += A_tile[t_x*tile_len + i] * B_tile[i*b_y + t_y];

        // wait for all threads to finish their computation before loading
        // the next tile
        __syncthreads();
    }

    C[a_i*b1 + b_j] = c;
}

__global__ void sum_axis(float *A, float *out, int axis, int a0, int a1)
{
    int a_i = blockDim.x*blockIdx.x + threadIdx.x;
    int start = 0;
    int stop = 0;
    int step = 0;
    if (axis == 0)
    {
        start = a_i;
        stop = a0*a1;
        step = a1;
    }
    else
    {
        start = a_i*a1;
        stop = start + a1;
        step = 1;
    }

    out[a_i] = 0;
    for (int i=start; i < stop; i += step)
        out[a_i] += A[i];
}

""" % {'SHARED_MEM': device.MAX_SHARED_MEMORY_PER_BLOCK})

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
    gpu_outer = mod.get_function("outer_sum")
    gpu_outer(a, b, out, batchsize,
              grid=(a_len / rows_per_block, b_len / cols_per_block),
              block=(rows_per_block, cols_per_block, 1))

    return out


def m_dot(a, b, out=None, transpose_b=False, increment=False):
    # not totally sure why this is necessary
    pycuda.autoinit.context.synchronize()

    a_shape = (np.int32(a.shape[0]), np.int32(a.shape[1]))

#     b_shape = (a_shape[1], b.size / a_shape[1])

    b_shape = (np.int32(b.shape[0]), np.int32(b.shape[1]))
    if transpose_b:
        b_shape = (b_shape[1], b_shape[0])

    assert a_shape[1] == b_shape[0]
    assert not increment or out is not None

    if out is None:
        out = gpuarray.zeros((a_shape[0], b_shape[1]), dtype=np.float32)

    assert type(a) == type(b) == type(out) == gpuarray.GPUArray

    # compute block size (we want blocks to be as square as possible so that
    # we get the most use of the data loaded into shared memory), and also
    # as large as possible so that we use all the threads
    if a_shape[0] * b_shape[1] < device.MAX_THREADS_PER_BLOCK:
        # simple case where we can fit everything in one block
        block_x = a_shape[0]
        block_y = b_shape[1]
    else:
        # start with the largest possible square block
        block_x = np.int32(np.sqrt(device.MAX_THREADS_PER_BLOCK))

        # make the block smaller until it evenly divides a_0
        while a_shape[0] % block_x != 0:
            block_x -= 1

        block_y = np.minimum(device.MAX_THREADS_PER_BLOCK / block_x,
                             b_shape[1])

        while b_shape[1] % block_y != 0:
            block_y -= 1

    # compute tile length (want tiles to be as large as possible while still
    # fitting in shared memory). the shared_mem attribute tells us how many
    # floats can fit in shared memory (bytes/4). so we want to pick the largest
    # tile_len such that tile_len * (block_x + block_y) < shared_mem
    tile_len = np.minimum(a_shape[1],
                          (device.MAX_SHARED_MEMORY_PER_BLOCK / 8) /
                          (block_x + block_y))
    while a_shape[1] % tile_len != 0:
        tile_len -= 1

#     print "multiplying", a.shape, b.shape, out.shape
#     print type(a), type(b), type(out)
#     print a.dtype, b.dtype, out.dtype
#     print "block", block_x, block_y
#     print "tile", tile_len

#     assert block_x * tile_len < device.MAX_SHARED_MEMORY_PER_BLOCK / 8
#     assert block_y * tile_len < device.MAX_SHARED_MEMORY_PER_BLOCK / 8

    gpu_m_dot = mod.get_function("m_dot")
    gpu_m_dot(a, b, out, a_shape[1], tile_len,
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

    gpu_sum_axis = mod.get_function("sum_axis")
    gpu_sum_axis(a, out, np.int32(axis), a_shape[0], a_shape[1],
                 grid=(a_shape[1 - axis] / block_len, 1),
                 block=(block_len, 1, 1))

    return out
