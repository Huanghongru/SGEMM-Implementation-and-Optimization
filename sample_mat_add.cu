#include "utils.cpp"
#include "blas.cu"
#include <stdio.h>

const int M = 100;
const int N = 100;
dim3 blocks(4, 4);
dim3 threads(16, 16);

__global__ void mat_add(double* a, double* b, double*c, int M, int N) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    for (int i = x; i < M; i += blockDim.x) {
        for (int j = y; j < N; j += blockDim.y) {
	    c[i*M+j] = a[i*M+j] + b[i*M+j];
	}
    }
}    

int main() {
    double* a = random_matrix_gpu<double>(M, N);
    double* b = random_matrix_gpu<double>(M, N);
    double* c = new double[M*N];
    double *dev_a, *dev_b, *dev_c;
   
    cudaMalloc((void**)&dev_a, M*N*sizeof(double));
    cudaMalloc((void**)&dev_b, M*N*sizeof(double));
    cudaMalloc((void**)&dev_c, M*N*sizeof(double));

    cudaMemcpy(dev_a, a, M*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, M*N*sizeof(double), cudaMemcpyHostToDevice);
    mat_add<<<blocks, threads>>>(dev_a, dev_b, dev_c, M, N);
    cudaMemcpy(c, dev_c, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    check_sum<double>(a, b, c, M, N);

    return 0;
}

