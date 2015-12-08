// #include <stdio.h>


__global__ void sum_cols_%float_type%(%float_type% *A, %float_type% *out, 
                                      const int increment,
                                      const int a0, const int a1)
{
    const int t_i = threadIdx.y;
    const int t_j = threadIdx.x;
    const int dim_i = blockDim.y;
    const int dim_j = blockDim.x;
    const int col = dim_j*blockIdx.x + t_j;
    const int A_offset = t_i*a1 + col;
    const int data_offset = t_i*dim_j + t_j;
    
    extern __shared__ float shared_data[];
    %float_type%* data = (%float_type%*)shared_data;
    
    // stage 1: loop threads across A to reduce to shared memory block
    const int step = dim_i*a1;
    const int limit = a0*a1;
    %float_type% sum = 0;
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


__global__ void iadd_%float_type%(%float_type% *A, %float_type% *v, 
                                  const int a0, const int a1)
{
    // in-place addition with broadcasting along first axis
    // (adding vector v to matrix A)
    
    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (row >= a0 || col >= a1)
        return;
    
    // load the appropriate part of v for this block into shared memory 
    __shared__ %float_type% v_share[32];
        
    if (threadIdx.y == 0)
        v_share[threadIdx.x] = v[col];
    
    __syncthreads();
    
    // add v to A
    A[row*a1 + col] += v_share[threadIdx.x];
    
}


__global__ void multiply_%float_type%(%float_type% *A, %float_type% *B, 
                                      %float_type% *out, 
                                      const int size, const int increment)
{
    // TODO: would it be faster to have each thread compute a couple entries?
    const int index = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (index >= size)
        return;
    
    if (increment)
        out[index] += A[index] * B[index];
    else
        out[index] = A[index] * B[index];
}


__global__ void arange(long long *out, const int size, const long long start0, 
                       const int inc0, const long long start1, const int inc1,
                       const long long start2, const int inc2)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (index >= size)
        return;
        
    if (blockIdx.y == 0)
        out[index] = start0 + index*inc0;
    else if(blockIdx.y == 1)
        out[index + size] = start1 + index*inc1;
    else
        out[index + (size<<1)] = start2 + index*inc2;
}
