#include <stdio.h>


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


__global__ void sum_cols(float *A, float *out, const int increment,
                         const int a0, const int a1)
{
    const int t_i = threadIdx.y;
    const int t_j = threadIdx.x;
    const int dim_i = blockDim.y;
    const int dim_j = blockDim.x;
    const int col = dim_j*blockIdx.x + t_j;
    const int A_offset = t_i*a1 + col;
    const int data_offset = t_i*dim_j + t_j;
    
    extern __shared__ float data[];
    
    // stage 1: loop threads across A to reduce to shared memory block
    const int step = dim_i*a1;
    const int limit = a0*a1;
    float sum = 0;
    int index = A_offset;
    for (int i=0; i < limit; i += step)
    {
        if (index < limit)
            sum += A[index];
        index += step;
    }
    data[data_offset] = sum;
    
    // stage 2: reduction within block
    // note: assumes that dim_i is divisible by 2
    for (int s=dim_i/2; s > 0; s>>=1)
    {
        __syncthreads();
        
        /*
        if (t_i == 0 && t_j == 0)
        {
            printf("data: ");
            for (int i=0; i < blockDim.x*blockDim.y; i++)
                printf("%f ", data[i]);
            printf("\n");
        }
        */
        
        if (t_i < s)
            data[data_offset] += data[data_offset + s*dim_j];
    }
    
    if (t_i == 0)
    {
        if (increment)
            out[col] += data[t_j];
        else
            out[col] = data[t_j];
    }
    
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

