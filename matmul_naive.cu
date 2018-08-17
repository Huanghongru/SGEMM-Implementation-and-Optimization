#include "utils.cpp"

dim3 threadsPerBlock(16, 16);

template <typename T>
__global__ void matmul_naive(T* a, T* b, T* c, int M, int K, int N) {
    /* A naive implementation of matrix multiplication.
     * a: MxK
     * b: KxN
     * c: MxN
     * 
     * Average Time: 1000x1000x1000, 4.85s
     * Average Time: 1024x1024x1024, 1.53s 
     */
    // If the whole threads can't cover the matrix elements,
    // the outside loop is required.
    // Here I only consider the case that the size of the matrix
    // is multiple of block size.
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    for (int i = x; i < M; i += blockDim.x) {
	for (int j = y; j < N; j += blockDim.y) {
	    c[i*M+j] = 0;
	    // A for loop in one thread caculates the 
	    // one value in output matrix.
	    for (int k = 0; k < K; ++k) {
		c[i*M+j] += a[i*M+k]*b[k*K+j];
	    }
	}
    }
}

int main(int argc, char *argv[]) {
    int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
    dim3 blocksPerGrid;
    blocksPerGrid.x = M / threadsPerBlock.x;
    blocksPerGrid.y = N / threadsPerBlock.y;
    blocksPerGrid.z = 1;

    double* a = utils::random_matrix_gpu<double>(M, K, utils::C_ORDER);
    double* b = utils::random_matrix_gpu<double>(K, N, utils::C_ORDER);
    double* c = new double[M*N];
    double *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, M*K*sizeof(double));
    cudaMalloc((void**)&dev_b, K*N*sizeof(double));
    cudaMalloc((void**)&dev_c, M*N*sizeof(double));

    cudaMemcpy(dev_a, a, M*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(double), cudaMemcpyHostToDevice);
    matmul_naive<double><<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, M, K, N);
    cudaMemcpy(c, dev_c, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << (utils::check_mul<double>(a, b, c, M, K, N, utils::C_ORDER)
		    ? "Correct!!" : "Wrong Answer!") << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}




