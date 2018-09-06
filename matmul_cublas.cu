#include "utils.cpp"
#include "cublas_v2.h"
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
    double *a = utils::random_matrix_gpu<double>(M, K, utils::FORTRAN_ORDER);
    double *b = utils::random_matrix_gpu<double>(K, N, utils::FORTRAN_ORDER);
    double *c = new double[M*N];

    double *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, M*K*sizeof(double));
    cudaMalloc((void**)&dev_b, K*N*sizeof(double));
    cudaMalloc((void**)&dev_c, M*N*sizeof(double));

    cudaMemcpy(dev_a, a, M*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(double), cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    double al=1.0f, bet=0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
		    &al, dev_a, M, dev_b, K, &bet, dev_c, M);

    cudaMemcpy(c, dev_c, M*N*sizeof(double), cudaMemcpyDeviceToHost);
#ifdef CHECK
    std::cout << (utils::check_mul<double>(a, b, c, M, K, N, utils::FORTRAN_ORDER) 
		    ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::FORTRAN_ORDER);
    std::cout << "\nMatrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::FORTRAN_ORDER);
    std::cout << "\nMatrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::FORTRAN_ORDER);
#endif

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
    
