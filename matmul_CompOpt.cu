#include "utils.cpp"

// Unlike Tiling, matrix B isn't need to be loaded into shared memory.
// We calculate the outer product of Asub and Bsub, where the size of
// Bsub is define by TILE_SIZE and VECTOR_SIZE. 
// Specifically: 
//   Asub: TILE_SIZE * TILE_SIZE 
//   Bsub: TILE_SIZE * (TILE_SIZE*VECTOR_SIZE)
const int TILE_SIZE = 16;
const int VECTOR_SIZE = 4;

template <typename T>
__global__ void matmul_CompOpt(T *A, T *B, T *C, int M, int K, int N) {
	/* Computation method optimization.
	 * Peform outer product instead of inner product to reduce  
	 * instructions from shared memory from two to one.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	// Explicitly allocate As as column-major array 
	// to store TILE*TILE submatrix of A.
	__shared__ T As[TILE_SIZE * TILE_SIZE];

	// Allocate register files at each thread.
	T cv[TILE_SIZE] = {0};

	// Basic iterations is similar with Tiling. But notice that 
	// the total number of threads is less than that of Tiling.
	int aBegin = K * TILE_SIZE * by;
	int aEnd = aBegin + K - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * VECTOR_SIZE * bx;
	int bStep = TILE_SIZE * N;

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Load Asub with size of TILE*TILE in colomn-major style.
		int t = TILE_SIZE / VECTOR_SIZE;
		for (int i = 0; i < t; ++i) {
			As[ (i*t+ty) + TILE_SIZE * tx] = A[a + K*(i*t+ty) + tx];
		}
		__syncthreads();

		T *ap = As;	// Point to the first address of As, increase later.
		// TODO: global memory ? register ? not clear :(
		T *bp = &B[b + TILE_SIZE * ty + tx];	

		for (int i = 0; i < TILE_SIZE; ++i) {
			T bv = *bp;	
		// Each thread calculate a vector of C with size of TILE_SIZE.
			for (int j = 0; j < TILE_SIZE; ++j) {
				cv[j] += ap[j] * bv;
			}
			ap += TILE_SIZE;
			bp += N;
		}
		__syncthreads();
	}
	
	// Store each value of Csub back to C in global memory.
	int c = N * TILE_SIZE * by + TILE_SIZE * VECTOR_SIZE * bx;
	c += TILE_SIZE * ty + tx;
	for (int i = 0; i < TILE_SIZE; ++i) {
		C[c] = cv[i];
		c += N;
	}
}

int main(int argc, char *argv[]) {
	int M = std::atoi(argv[1]);
	int K = std::atoi(argv[2]);
	int N = std::atoi(argv[3]);

	dim3 threads(TILE_SIZE, VECTOR_SIZE);
	dim3 grid(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);

	double *a = utils::random_matrix_gpu<double>(M, K, utils::C_ORDER);
	double *b = utils::random_matrix_gpu<double>(K, N, utils::C_ORDER);
	double *c = new double[M*N];

	double *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, M*K*sizeof(double));
	cudaMalloc((void**)&dev_b, K*N*sizeof(double));
	cudaMalloc((void**)&dev_c, M*N*sizeof(double));

	cudaMemcpy(dev_a, a, M*K*sizeof(double), cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_b, b, K*N*sizeof(double), cudaMemcpyHostToDevice);
	
	matmul_CompOpt<double><<<grid, threads>>>(dev_a, dev_b, dev_c, M, K, N);

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


