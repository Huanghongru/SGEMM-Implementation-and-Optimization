#include "../utils.cpp"
#include <stdio.h>

__global__ void add (int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
	c[tid] = a[tid]+b[tid];
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
    int a[N], b[N], c[N], c_[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
	    c_[i] = a[i] + b[i];
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N / TILE_SIZE, TILE_SIZE>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
	printf("Naive addition:");
    printf(check_sum(c, c_, N) ? "Correct!\n" : "Wrong...\n");

#ifdef DEBUG
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c_op[i]); 
    }std::cout << std::endl;
#endif

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

