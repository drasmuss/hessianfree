#include <stdio.h>


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

__global__ void sum_axis(float *A, float *out, const int axis, const int a0, 
                         const int a1, const int increment)
{
    // sum matrix A over the specified axis
    // TODO: replace this with a reduction kernel
    
	int a_i = blockDim.x*blockIdx.x + threadIdx.x;
	int start = 0;
	int stop = 0;
	int step = 0;
	if (axis == 0)
	{
	    if (a_i >= a1)
	       return;
		start = a_i;
		stop = a0*a1;
		step = a1;
	}
	else
	{
	    if (a_i >= a0)
	       return;
		start = a_i*a1;
		stop = start + a1;
		step = 1;
	}

	float sum = 0;
	for (int i = start; i < stop; i += step)
		sum += A[i];
	
	if (increment)
	   out[a_i] += sum;
	else
	   out[a_i] = sum;
}

__global__ void iadd(float *A, float *v, const int a0, const int a1)
{
    // in-place addition with broadcasting along first axis
    // (adding vector v to matrix A)
    
    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (row >= a0 || col >= a1)
        return;
    
    // load the appropriate part of v for this block into shared memory
    // TODO: check if this shared memory is faster than just L1 caching   
    __shared__ float v_share[32];
        
    if (threadIdx.y == 0)
        v_share[threadIdx.x] = v[col];
    
    __syncthreads();
    
    // add v to A
    A[row*a1 + col] += v_share[threadIdx.x];
    
}

