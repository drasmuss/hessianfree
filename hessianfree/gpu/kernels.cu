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
    for (int j = 0; j < batch_size; j++) 
    {
        out[out_addr] += a[a_i] * b[b_i];
        a_i += a_len;
        b_i += b_len;
	}

	// TODO: convert this to a tiled approach like m_dot
}


__global__ void simple_m_dot(float *A, float *B, float *C, int a1,
	int transpose_b, int increment)
{
	// matrix multiplication without any of the shared memory tiling

	const int b1 = blockDim.y*gridDim.y;

	// row/col for this thread
	const int a_i = blockIdx.x*blockDim.x + threadIdx.x;
	const int b_j = blockIdx.y*blockDim.y + threadIdx.y;

	// offsets for each matrix
	const int A_off = a_i * a1;
	const int A_end = A_off + a1;
	int B_off = b_j;
	int B_step = b1;
	if (transpose_b)
	{
		B_off = b_j * a1;
		B_step = 1;
	}
	const int C_off = a_i*b1 + b_j;

	float c;
	if (increment)
		c = C[C_off];
	else
		c = 0;

	// accumulate the product for this thread
	for (int i = A_off, j = B_off; i < A_end; i++, j += B_step)
		c += A[i] * B[j];

	/*
	if ((a_i == 0 && b_j == 0) || (a_i == 7499 && b_j == 0) ||
	(a_i == 0 && b_j == 1023) || (a_i == 7499 && b_j == 1023))
	{
	printf("data for %%d %%d\\n", a_i, b_j);
	printf("a1 %%d b1 %%d\\n", a1, b1);
	printf("Aoff %%d Aend %%d i %%d\\n", A_off, A_end, i);
	printf("Boff %%d Bstep %%d j %%d\\n", B_off, B_step, j);
	}
	*/

	C[C_off] = c;
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
	for (int i = start; i < stop; i += step)
		out[a_i] += A[i];
}