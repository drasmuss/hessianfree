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


__global__ void shared_m_dot_%float_type%_%transpose_a%_%transpose_b%(
        %float_type% *A, %float_type% *B, %float_type% *C, const int a0, 
        const int a1, const int b1, const int increment)
{
    // multiplying an [a0,a1] matrix by an [a1,b1] matrix. each thread will
    // compute one element of c, which is a[i,:] * b[:,j].  however, we can
    // make this more efficient by e.g. loading a[i,:] into shared memory, and
    // then computing all the c[i,:] elements off that. we probably can't fit
    // all the necessary data into shared memory, so we load in a[i,:tile_len]
    // and b[:tile_len, j] sized chunks one at a time.
    
    // this doesn't necessarily need to be the case, but it seems to be most 
    // efficient to make the tiles square (tile_len = blockDim.x = blockDim.y)
    
    // each thread is responsible for loading one cell from a and one cell
    // from b into shared memory.  however, note that the cells a thread
    // loads into memory are not necessarily tied to the rows/cols it needs
    // to multiply to compute c[i,j] (they are disconnected so that we can
    // make sure that the memory loading is always occurring in the most
    // efficient way possible).
    
    // side length of square tile
    const int tile_len = blockDim.x;
    
    // thread variables
    // note: we use x to index cols and y to index rows because warps will be
    // clustered along x, which we want to be memory aligned to promote
    // coalescing
    const int t_i = threadIdx.y;
    const int t_j = threadIdx.x;
    
    // row/col for this block
    const int block_i = blockIdx.y*blockDim.y;
    const int block_j = blockIdx.x*blockDim.x;
    
    // row/col for this thread
    const int a_i = block_i + t_i;
    const int b_j = block_j + t_j;
    
    // if this thread is involved in computing a c[i,j] entry (it can still be
    // involved in loading data even if this is false)
    const bool active_c = a_i < a0 && b_j < b1;
    
    #if %transpose_a%
        // whether this thread is involved in loading data for the A tile
        const bool active_a = block_i + t_j < a0;
        
        // the index where this thread will load from
        const int A_off = block_i + t_i*a0 + t_j; 
        
        // the distance along the tile axis that this thread loads from
        const int A_axis_off = t_i;
    #else
        const bool active_a = block_i + t_i < a0;
        const int A_off = (block_i + t_i)*a1 + t_j;
        const int A_axis_off = t_j;
    #endif
                                     
    #if %transpose_b%                     
        const bool active_b = block_j + t_i < b1;
        const int B_off = (block_j + t_i)*a1 + t_j;
        const int B_axis_off = t_j;
    #else
        const bool active_b = block_j + t_j < b1;
        const int B_off = block_j + t_i*b1 + t_j;
        const int B_axis_off = t_i;
    #endif
    
    // the index where this thread puts its data in the tile
    const int tile_off = t_i*(tile_len+1) + t_j;
    
    #if %transpose_a%
        // loop variables for outer loop (across tiles)
        const int outer_A_step = a0*tile_len;
        
        // loop variables for inner loop (within tile)
        const int inner_A_start = t_i;
        const int inner_A_step = tile_len+1;
    #else
        const int outer_A_step = tile_len;
        const int inner_A_start = t_i*(tile_len+1);
        const int inner_A_step = 1;
    #endif
    #if %transpose_b%
        const int outer_B_step = tile_len;
        const int inner_B_start = t_j*(tile_len+1);                   
        const int inner_B_step = 1;
    #else
        const int outer_B_step = b1*tile_len;
        const int inner_B_start = t_j;                   
        const int inner_B_step = tile_len+1;
    #endif

    // c will accumulate the value of c[i,j] across tiles
    const int C_off = a_i*b1 + b_j;
    %float_type% c = 0;

    /*
    if (a_i == 1 && b_j == 1)
    {
        printf("a_i %d, b_j %d \n", a_i, b_j);
        printf("transpose_b %d, increment %d \n", transpose_b, increment);
        printf("A_block_off %d, A_limit %d, outer_A_step %d \n", A_block_off, 
               A_limit, tile_len);
        printf("A_off %d \n", A_off);
        printf("B_block_off %d, B_limit %d, outer_B_step %d \n", B_block_off, 
               B_limit, outer_B_step);
        printf("B_off %d \n", B_off);
        printf("tile_off %d \n", tile_off);
        printf("inner_A_start %d, inner_A_end %d \n", inner_A_start, 
               inner_A_end);
        printf("inner_B_start %d, inner_B_step %d \n", inner_B_start, 
               inner_B_step);
    }
    */
    
    // create the tiles
    // note: we add an extra column (which will just be zero) so that
    // when we are writing data in by column the writes are all offset,
    // reducing bank conflicts
    extern __shared__ float shared_data[];
    %float_type%* A_tile = (%float_type%*)shared_data;
    %float_type%* B_tile = A_tile + (tile_len+1)*blockDim.y;
    
    // loop over the tiles
    // tile_i and tile_j point to the top left corner of the A/B tiles
    int tile_i = A_off;
    int tile_j = B_off;
    for (int tile=0; tile < a1; tile+=tile_len)
    {
        // each thread loads in its part of A/B
        // we need to check whether the location of this thread in the current
        // tile extends past the rows/cols of A and B, for the case when those
        // rows/cols are not evenly divisible by block len
        // TODO: we could create an "unsafe" version of this kernel that 
        // assumes everything is evenly divisible by block len
        
        if (active_a && tile + A_axis_off < a1)
            A_tile[tile_off] = A[tile_i];
        else
            A_tile[tile_off] = 0;
        
        if (active_b && tile + B_axis_off < a1)
            B_tile[tile_off] = B[tile_j];
        else
            B_tile[tile_off] = 0;

        // wait for all threads to finish loading in their data
        __syncthreads();
        
        /*
        if (a_i == 1 && b_j == 1)
        {
            printf("tile_i %d, tile_j %d \n", tile_i, tile_j);
            printf("tile_off %d \n", tile_off);
            printf(" %f %f \n", A[tile_i + A_off], B[tile_j + B_off]);
            printf(" %f %f \n", A_tile[tile_off], B_tile[tile_off]);
            for(int i=0; i < blockDim.x * tile_len; i++)
                printf("%f ", A_tile[i]);
            printf("\n");
            for(int i=0; i < blockDim.y * tile_len; i++)
                printf("%f ", B_tile[i]);
            printf("\n");
            printf("\n");
        }
        */
        
        // accumulate the product for this thread
        if (active_c)
        {
            int A_index = inner_A_start;
            int B_index = inner_B_start;
            for (int i = 0; i < tile_len; i++)
            {
                c += A_tile[A_index] * B_tile[B_index];

                /*
                if (a_i == 1 && b_j == 1)
                {
                    printf("%d %d \n", i, j);
                    printf("%f * %f, %f \n", A_tile[i], B_tile[j], c);
                }
                */
                
                A_index += inner_A_step;
                B_index += inner_B_step;
            }
        }
    
        tile_i += outer_A_step;
        tile_j += outer_B_step;
        
        // wait for all threads to finish their computation before loading
        // the next tile
        __syncthreads();
    }
    
    if (active_c && increment)
        C[C_off] += c;
    else if (active_c)
        C[C_off] = c;
}


__global__ void mv_batched_%float_type%_%transpose_a%(
        %float_type% *A, %float_type% *v, %float_type% *out, 
        const int a0, const int a1, const int increment)
{
    // batched matrix-vector product
    
    const int t_i = threadIdx.y;
    const int t_j = threadIdx.x;
    const int dim_i = blockDim.y;
    const int dim_j = blockDim.x;
    
    // note: right now this code assumes that dim_i == dim_j
    // it also assumes that dim_i is evenly divisible by 2
    
    // batch offset
    A += blockIdx.y * a0 * a1;
    v += blockIdx.y * a1;
    out += blockIdx.y * a0;
    
    extern __shared__ float shared_data[];
    %float_type%* data = (%float_type%*)shared_data;
    %float_type%* v_share = data + dim_i*(dim_j+1);
    
    #if %transpose_a%
        const int start = dim_j*blockIdx.x;
        const int step = dim_i;
        const int offset_step = step*a0;
        const int limit = a0*a1;
        const int v_index = t_i;
        const int data_offset = t_i*(dim_j+1) + t_j;
        const bool active = dim_j*blockIdx.x + t_j < a0;
        const int block_offset = t_i*a0 + t_j;
    #else
        const int start = dim_i*blockIdx.x*a1;
        const int step = dim_j;
        const int offset_step = step;
        const int limit = start + (t_i+1)*a1;
        const int v_index = t_j;
        const int data_offset = t_j*(dim_j+1) + t_i;
        const bool active = dim_i*blockIdx.x + t_i < a0;
        const int block_offset = t_i*a1 + t_j;
    #endif
    
    %float_type% sum = 0;
    int block_index = start + block_offset;
    for (int i=0; i < a1; i+=step)
    {
        if (t_i == 0 && i + t_j < a1)
            v_share[t_j] = v[i + t_j];
        
        __syncthreads();
        
        if (active && block_index < limit)
            sum += A[block_index] * v_share[v_index];
        block_index += offset_step;
    }
    data[data_offset] = sum;
    
    /*
    if (blockIdx.y == 3 && t_i == 15 && t_j == 0)
    {
        printf("v:");
        for (int i=0; i < 32; i++)
            printf(" %f", v[i]);
        printf("\n");
        
        printf("v_share:");
        for (int i=0; i < 32; i++)
            printf(" %f", v_share[i]);
        printf("\n");
        
        printf("data:");
        for (int i=0; i < 1024; i++)
            printf(" %f", data[i]);
        printf("\n");
    }
    */
    
    // stage 2: reduction within block
    // note: we always order things in the block such that we can do a 
    // column-wise reduction (to keep warps intact)
    const int out_offset = blockIdx.x*dim_j + t_j;
    if (out_offset >= a0)
        return;
    
    const int reduction_offset = t_i*(dim_j+1) + t_j;
    for (int s=dim_i/2; s > 0; s>>=1)
    {
        __syncthreads();
        
        /*
        if (t_i == 0 && t_j == 0)
        {
            printf("data:");
            for (int i=0; i < 32; i++)
                printf(" %f", data[i]);
            printf("\n");
        }
        */
        
        if (t_i < s)
            data[reduction_offset] += data[reduction_offset + s*(dim_j+1)];
    }

    if (increment && t_i == 0)
        out[out_offset] += data[t_j];
    else if (t_i == 0)
        out[out_offset] = data[t_j];
}
