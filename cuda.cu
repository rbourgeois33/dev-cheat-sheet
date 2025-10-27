// Warp level primitive with human readable name

template <typename T>
__inline__ __device__
T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__
bool warpAllTrue(bool predicate){
    return __all_sync(0xffffffff, predicate);
}

__inline__ __device__
unsigned warpWhichAreTrue(bool predicate){
    return __ballot_sync(0xffffffff, predicate);
}

__inline__ __device__
int warpMaxIndexTrue(bool predicate){

    unsigned mask =  warpWhichAreTrue(predicate);
    return (mask != 0) ? (31 - __clz(mask)) : -1;
}

void inline __device__ vectorized_store_from_smem_to_gmem(int DLB_blockIdx, raft::device_span<int> buffer, int *sdata, int size, int prefix){
    
    assert(WPT%4==0); //Obligé pour la vecto
    int global_thread_base = DLB_blockIdx * WPT4 * BLOCK_SIZE + threadIdx.x;

    int4* buffer_vec = reinterpret_cast<int4*>(buffer.data());
    int4* sdata_vec = reinterpret_cast<int4*>(sdata);

    // Write back scanned data from smem to gmem and adding prefix
    if (DLB_blockIdx==blockDim.x-1){  //Only last block needs to be careful
        #pragma unroll
        for (int k = 0; k < WPT4; k++) {
            const int thread_offset = k*BLOCK_SIZE;
            const int i_global = global_thread_base + thread_offset;
            const int i_local = threadIdx.x + thread_offset;
            
            if ((i_global * 4 + 3) < size) {
                // Fully within bounds - vectorized load, add prefix, vectorized store             
                int4 temp = sdata_vec[i_local];
                temp.x += prefix;
                temp.y += prefix;
                temp.z += prefix;
                temp.w += prefix;
                buffer_vec[i_global] = temp;
            } else {
                // Handle boundary case - scalar operations
                int base_idx = i_global * 4;
                if (base_idx + 0 < size) buffer[base_idx + 0] = sdata[base_idx + 0] + prefix;
                if (base_idx + 1 < size) buffer[base_idx + 1] = sdata[base_idx + 1] + prefix;
                if (base_idx + 2 < size) buffer[base_idx + 2] = sdata[base_idx + 2] + prefix;
                if (base_idx + 3 < size) buffer[base_idx + 3] = sdata[base_idx + 3] + prefix;
            }
        }
    } 
    else {
        #pragma unroll
        for (int k = 0; k < WPT4; k++) {
            const int thread_offset = k*BLOCK_SIZE;
            const int i_global = global_thread_base + thread_offset;
            const int i_local = threadIdx.x + thread_offset;
            
            int4 temp = sdata_vec[i_local];
            temp.x += prefix;
            temp.y += prefix;
            temp.z += prefix;
            temp.w += prefix;
            buffer_vec[i_global] = temp;
        }
    }
}


  int thread_index_within_warp = threadIdx.x & 31;        // Same as threadIdx.x % 32 (faster with bitwise AND)
    int warp_id = threadIdx.x >> 5;        // Same as threadIdx.x / 32 (faster with bit shift)