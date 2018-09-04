#include "utils.cpp"

const int TILE_SIZE = 16;

template <typename T>
__global__ void matmul_Tiling(T *A, T *B, T *C, int M, int K, int N) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ T As[TILE_SIZE][TILE_SIZE];
	__shared__ T Bs[TILE_SIZE][TILE_SIZE];

	int aBegin = K * TILE_SIZE * by;
	int aEnd = aBegin + K - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * bx;
	int bStep = TILE_SIZE * N;

	T Csub = 0;

	for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
		As[ty][tx] = A[i + K * ty + tx];
		Bs[tx][ty] = B[j + N * tx + ty];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub += As[ty][k]*Bs[k][tx];
		}
		
		__syncthreads();
	}
	int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
	C[cIdx + N * ty + tx] = Csub;
}

int main(int argc, char *argv[]) {
	int M = std::atoi(argv[1]);
	int K = std::atoi(argv[2]);
	int N = std::atoi(argv[3]);

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid(N / TILE_SIZE, M / TILE_SIZE);

	double *a = utils::random_matrix_gpu<double>(M, K, utils::C_ORDER);
	double *b = utils::random_matrix_gpu<double>(K, N, utils::C_ORDER);
	double *c = new double[M*N];
	
	double *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, M*K*sizeof(double));
	cudaMalloc((void**)&dev_b, K*N*sizeof(double));
	cudaMalloc((void**)&dev_c, M*N*sizeof(double));

	cudaMemcpy(dev_a, a, M*K*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, K*N*sizeof(double), cudaMemcpyHostToDevice);

	matmul_Tiling<double><<<grid, threads>>>(dev_a, dev_b, dev_c, M, K, N);

	cudaMemcpy(c, dev_c, M*N*sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << (utils::check_mul<double>(a, b, c, M, K, N, utils::C_ORDER) ? "Correct!!" : "Wrong Answer!") << std::endl;

#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::C_ORDER);
    std::cout << "Matrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::C_ORDER);
    std::cout << "Matrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::C_ORDER);
#endif
	return 0;
}

