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