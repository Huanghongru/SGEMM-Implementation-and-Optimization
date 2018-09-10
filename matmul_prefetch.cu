#include "utils.cpp"

// Prefetching is a further improvment of computational optimization method.
// This method is based on the parallelism of load and stroe instruction
// on GPU device.

const int TILE_SIZE = 16;
const int VECTOR_SIZE = 4;

template <typename T>
__global__ void matmul_Prefetch(T *A, T *B, T *C, int M, int K, int N) {
	/* Prefetching method.
	 * Perform outer product of Asub and Bsub.
	 * Specifically:
	 *   Asub: TILE_SIZE * TILE_SIZE
	 *   Bsub: TILE_SIZE * (TILE_SIZE * VECTOR_SIZE)
	 * 
	 * Before calculating the submatrix, load the next TILE * TILE
	 * submatrix of A into register.
	 *
	 * After calculating, just swap the pointer to exchange the submatrix.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	// Allocate As and next_As as column-major array
	__shared__ T As[TILE_SIZE * TILE_SIZE];
	__shared__ T next_As[TILE_SIZE * TILE_SIZE];

	// Allocate register files for sub-result of C at each thread.
	T cv[TILE_SIZE] = {0};

	// Iteration parameters is similar with 
	// computational optimization method.
	int aBegin = K * TILE_SIZE * by;
	int aEnd = aBegin + K - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * VECTOR_SIZE * bx;
	int bStep = TILE_SIZE * N;

	int t = VECTOR_SIZE;
	T *cur = As;
	T *nxt = next_As;
	for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i) {
		cur[ (i*t+ty) + TILE_SIZE * tx] = A[aBegin + K*(i*t+ty) + tx];
	}
	__syncthreads();

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Load the next submatrix to another register files.
		// Should check the out-of-range indexing to avoid kernel crash.
		if (a+aStep <= aEnd) {
		    for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i) {
				nxt[ (i*t)+ty + TILE_SIZE * tx] = A[a + K*(i*t+ty) + tx + aStep];
			}
		}
		T *ap = cur;
		T *bp = &B[b + TILE_SIZE * ty + tx];

		for (int i = 0; i < TILE_SIZE; ++i) {
			T bv = *bp;
			for (int j = 0; j < TILE_SIZE; ++j) {
				cv[j] += ap[j] * bv;
			}
			ap += TILE_SIZE;
			bp += N;
		}
		__syncthreads();

		// Swap current submatrix and next submatrix.
		// Note that you can't directly assign nxt to cur, which
		// will change cur and nxt simultaneously at the next loop.
		T *tmp = cur;
		cur = nxt;
		nxt = tmp;
	}

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

	float *a = utils::random_matrix_gpu<float>(M, K, utils::C_ORDER);
	float *b = utils::random_matrix_gpu<float>(K, N, utils::C_ORDER);
	float *c = new float[M*N];

	float *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, M*K*sizeof(float));
	cudaMalloc((void**)&dev_b, K*N*sizeof(float));
	cudaMalloc((void**)&dev_c, M*N*sizeof(float));

	cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);
	
	matmul_Prefetch<float><<<grid, threads>>>(dev_a, dev_b, dev_c, M, K, N);

	cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
#ifdef CHECK
	std::cout << (utils::check_mul<float>(a, b, c, M, K, N, utils::C_ORDER) ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::C_ORDER);
    std::cout << "Matrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::C_ORDER);
    std::cout << "Matrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::C_ORDER);
#endif
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	free(a);
	free(b);
	free(c);
	return 0;
}




