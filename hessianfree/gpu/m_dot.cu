__global__ void shared_m_dot_%transpose_a%_%transpose_b%(
        float *A, float *B, float *C, int a0, int a1, int b1, int increment)
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
    #define tile_len %tile_len%
    
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
    
    // is this thread involved in computing a c[i,j] entry (it can still be
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
    float c = 0;

    /*
    if (a_i == 1 && b_j == 1)
    {
        printf("a_i %d, b_j %d \n", a_i, b_j);
        printf("transpose_b %d, increment %d \n", transpose_b, increment);
        printf("A_block_off %d, A_limit %d, outer_A_step %d \n", A_block_off, A_limit, tile_len);
        printf("A_off %d \n", A_off);
        printf("B_block_off %d, B_limit %d, outer_B_step %d \n", B_block_off, B_limit, outer_B_step);
        printf("B_off %d \n", B_off);
        printf("tile_off %d \n", tile_off);
        printf("inner_A_start %d, inner_A_end %d \n", inner_A_start, inner_A_end);
        printf("inner_B_start %d, inner_B_step %d \n", inner_B_start, inner_B_step);
    }
    */
    
    // create the tiles
    // note: we add an extra column (which will just be zero) so that
    // when we are writing data in by column the writes are all offset,
    // reducing bank conflicts
    __shared__ float A_tile[1056]; //[tile_len][tile_len+1];
    __shared__ float B_tile[1056]; //[tile_len][tile_len+1];
    
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
            for (int i = 0; i < tile_len; i++)
            {
                c += (A_tile[inner_A_start + i*inner_A_step] * 
                      B_tile[inner_B_start + i*inner_B_step]);

                /*
                if (a_i == 1 && b_j == 1)
                {
                    printf("%d %d \n", i, j);
                    printf("%f * %f, %f \n", A_tile[i], B_tile[j], c);
                }
                */
            }
        }
    
        // wait for all threads to finish their computation before loading
        // the next tile
        __syncthreads();
        
        tile_i += outer_A_step;
        tile_j += outer_B_step;
    }
    
    if (active_c)
    {
        if (increment)
            C[C_off] += c;
        else
            C[C_off] = c;
    }
}
