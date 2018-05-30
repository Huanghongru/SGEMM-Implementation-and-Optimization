#include "utils.cpp"
#include "blas.cu"
#include <stdio.h>

#define N 10

/* CUDA offers a relatively light-weight alternative to
     * CPU timers via the CUDA event API
     * The logic of usage is as follows.
     */

int main() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    // Create CUDA event obeject by cudaEvent_t
    // Use cudaEventCreate API to create    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i*i;
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    // Squeeze the function to time by invoke API
    // cudaEventRecord twice.
    // Kernel call is asynchronized to host code.
    cudaEventRecord(start);
    add<<<N, 1>>>(dev_a, dev_b, dev_c, N);
    cudaEventRecord(stop);

    // Need to block CPU execution until 'stop' is record, which
    // means the execution on GPU is completed.
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]); 
    }std::cout << std::endl;
   
    // Refer a float type variable to calculate
    // the elapsed time. 
    float t = 0;
    cudaEventElapsedTime(&t, start, stop);
    printf("add takes %f ms to complete\n", t);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

