#include "utils.cpp"
#include "blas.cu"
#include <stdio.h>

#define N 100000

__global__ void large_scale_add(int *a, int *b, int *c) {
    /* For 100,000 elements, this functions takes about 0.01ms
     * to complete.
     */
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += gridDim.x*blockDim.x;
    }
}

int main() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

   
    cudaEventRecord(start);
    large_scale_add<<<256, 256>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop);
 
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    /* 
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]); 
    }std::cout << std::endl;*/
   
    float t = 0;
    cudaEventElapsedTime(&t, start, stop);
    printf("add takes %f ms to complete\n", t);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

