#include "../utils.cpp"
#include <stdio.h>

__global__ void add_op(int *a, int *b, int *c, int TILE_SIZE, int threadSize) {
    int tid = threadIdx.x + blockIdx.x*(blockDim.x*(TILE_SIZE/threadSize));
#pragma unroll
	for (int i = 0; i < TILE_SIZE/threadSize; ++i) {
		c[tid+i*threadSize] = a[tid+i*threadSize]+b[tid+i*threadSize];
	}
}

bool check_sum(int *c, int *c_, int n) {
    for (int i = 0; i < n; ++i) {
	if (c[i] != c_[i])
	    return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    const int N = std::atoi(argv[1]);
    const int TILE_SIZE = 16;
    int a[N], b[N], c_[N], c_op[N];
    int *dev_a, *dev_b, *dev_c_op;

    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c_op, N*sizeof(int));

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
	    c_[i] = a[i] + b[i];
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	int threadSize = TILE_SIZE / 2;
	add_op<<<N / TILE_SIZE, threadSize>>>(dev_a, dev_b, dev_c_op, TILE_SIZE, threadSize);

	cudaMemcpy(c_op, dev_c_op, N*sizeof(int), cudaMemcpyDeviceToHost);
	printf("Optimized addition:");
	printf(check_sum(c_op, c_, N) ? "Correct!\n" : "Wrong...\n");

#ifdef DEBUG
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c_op[i]); 
    }std::cout << std::endl;
#endif

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c_op);

    return 0;
}

