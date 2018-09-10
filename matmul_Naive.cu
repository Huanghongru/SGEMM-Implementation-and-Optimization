#include "utils.cpp"

const int TILE_SIZE = 16;

template <typename T>
__global__ void matmul_naive(T* a, T* b, T* c, int M, int K, int N) {
    /* A naive implementation of matrix multiplication.
     * a: MxK
     * b: KxN
     * c: MxN
     */
    // If the whole threads can't cover the matrix elements,
    // the outside loop is required.
    // Here I only consider the case that the size of the matrix
    // is multiple of block size.
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    int j = threadIdx.x + blockIdx.x*blockDim.x;

	// A for loop in one thread caculates the 
	// one value in output matrix.
	T elem = 0;
	for (int k = 0; k < K; ++k) {
		elem = elem + a[i*K+k]*b[k*N+j];
	}
	c[i*N+j] = elem;
}

int main(int argc, char *argv[]) {
    int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, M / TILE_SIZE);

    float* a = utils::random_matrix_gpu<float>(M, K, utils::C_ORDER);
    float* b = utils::random_matrix_gpu<float>(K, N, utils::C_ORDER);
    float* c = new float[M*N];
    float *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, M*K*sizeof(float));
    cudaMalloc((void**)&dev_b, K*N*sizeof(float));
    cudaMalloc((void**)&dev_c, M*N*sizeof(float));

    cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);
    matmul_naive<float><<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, M, K, N);
    cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

#ifdef CHECK
    std::cout << (utils::check_mul<float>(a, b, c, M, K, N, utils::C_ORDER)
		    ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}




