#include "../utils.cpp"
#include <stdio.h>

const int N = 1000;
const int blocksPerGrid = 256;
const int threadsPerBlock = 256;

__global__ void dev_dot(int *a, int *b, int *c) {
    /* This function returns a partial result of a*b^T.
     * Because the number of the elements in partial 
     * result is quite small so that it is more suitable
     * for CPU to complete.
     *
     * To simplify, this function only applies to vectos
     * with dimension greater than 256...
     */ 

    // The compiler will create a copy of the shared variables
    // for each block.
    __shared__ int cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x; 
    int temp = 0;
    while (tid < N) {
        temp += a[tid]*b[tid];
        tid += blockDim.x * gridDim.x;
    }
    // Set the cache values for current thread.
    cache[cacheIndex] = temp;

    // Synchronize threads in this block
    // This call guarantees that every thread in the block has
    // completed instructions prior to the __syncthreads() before
    // the hardware will execute the next instruction on any thread.
    __syncthreads();
    
    // for reductions, threadsPerBlock must be a power of 2
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        // Make sure all threads have completed this reduction procedure.
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex==0)
        c[blockIdx.x] = cache[0];
}

int dot(int*a, int*b) {
    /* A wrapper of CUDA dot function.
     */
    int *dev_a, *dev_b, *dev_c, c[N];
    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dev_dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);
  
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    int result = 0;
    for (int i = 0; i < blocksPerGrid; ++i)
        result += c[i];

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return result;
}

int main() {
    int a[N], b[N];
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    } 
   
    cudaEventRecord(start);
    int dot_res = dot(a, b);
    cudaEventRecord(stop);
 
    cudaEventSynchronize(stop);
    /* 
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]); 
    }std::cout << std::endl;*/
    std::cout << dot_res << std::endl;
   
    float t = 0;
    cudaEventElapsedTime(&t, start, stop);
    printf("add takes %f ms to complete\n", t);

    return 0;
}

