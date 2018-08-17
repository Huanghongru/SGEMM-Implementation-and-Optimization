#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>

namespace utils {
int C_ORDER = 1;
int FORTRAN_ORDER = 2;

template <typename T>
T** random_fill_matrix(int row, int col, T min=0, T max=100) {
    /* A function to quickly generate a matrix in some range [min, max]
     * Parameters:
     *   row: number of rows of matrix
     *   col: number of columns of matrix
     *   min, max: the range of random number. default to [0, 100]
     * Returns:
     *   a specific type 2d pointer pointed to the matrix
     */
    T** mat = new T*[row];
    for (int i = 0; i < col; ++i) {
        mat[i] = new T[col];
    }

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> unif(min, max);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
	    mat[i][j] = unif(mt);
        }
    }
    return mat;
}

template <typename T>
T* random_matrix_gpu(int row, int col, int order_type, T min=-50, T max=50) {
    /* A function to quickly generate a matrix in some range [min, max)
     * Note that it is very hard to allocate 2-d array on GPU,
     * so in most of the cases, we pass the 2-d array as a 1-d array
     * to the device following row-major or column-major order.
     *
     * Parameters:
     * ----------
     *   row: number of rows of matrix
     *   col: number of columns of matrix
     *   min, max: the range of random number. default to [-50, 50)
     *
     * Returns:
     * -------
     *    a specific type 1d pinter pointed to the matrix
     */
    T* mat = new T[row*col];
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> unif(min, max);
    if (order_type == C_ORDER ) {
        for (int i = 0; i < row; ++i) {
	    for (int j = 0; j < col; ++j) {
	        mat[i*row+j] = unif(mt);
	    }
        }
    } else {
        for (int i = 0; i < row; ++i) {
	    for (int j = 0; j < col; ++j) {
	        mat[i+j*row] = unif(mt);
	    }
        }
    }
    return mat;
}

template <typename T>
void print_mat_gpu(T* mat, int row, int col, int order_type) {
    if (order_type == C_ORDER) {
        for (int i = 0; i < row; ++i) {
	    for (int j = 0; j < col; ++j) {
	        std::cout << mat[i*row + j] << " ";
	    }std::cout << std::endl;
        }
    } else {
        for (int i = 0; i < row; ++i) {
	    for (int j = 0; j < col; ++j) {
	        std::cout << mat[i + j*row] << " ";
	    }std::cout << std::endl;
        }
    }
}

template <typename T>
void print_mat(T** mat, int row, int col) {
    // Display the matrix for visualizatoin
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::cout << mat[i][j] << " ";
        }std::cout << std::endl;
    }
}

template <typename T>
void check_sum(T* a, T* b, T* c, int row, int col) {
    bool flag = true;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
	    if (a[i*row+j]+b[i*row+j] != c[i*row+j]) {
		flag = 0;
		std::cout << "Wrong Answer!!!" << std::endl;
		break;
	    }
	    
            std::cout << a[i*row+j] << " + " << b[i*row+j]
		    << " = " << c[i*row+j] << " "
		    << bool(a[i*row+j]+b[i*row+j]==c[i*row+j]) << std::endl;
	}
	if (!flag) break;
    }
}

template <typename T>
bool check_mul(T* a, T* b, T* c, int M, int K, int N, int order_type) {
    /* Check if the result of matrix multiplication is right.*/
    if (order_type == C_ORDER) {
	for (int i = 0; i < M; ++i) {
	    for (int j = 0; j < N; ++j) {
		T value = 0;
		for (int k = 0; k < K; ++k) {
		    value += a[i*M+k]*b[k*K+j];
		}
		if (fabs(value-c[i*M+j])>1e-5) {
		    std::cout << c[i*M+j] << " " << value << std::endl;
		    return false;
		}
	    }
	}
    } else {
	for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
		T value = 0;
		for (int k = 0; k < K; ++k) {
		    value += a[i+k*M]*b[k+j*K];
		}
		if (fabs(value-c[i+j*M])>1e-5) {
		    std::cout << c[i+j*M] << " " << value << std::endl;
		    return false;
		}
	    }
	}
    }
    return true;
}
}
/*
int main() {
    int a[4] = {1, 2, 3, 4};
    int b[4] = {1, 2, 3, 4};
    int c[4] = {6, 10, 15, 22};
    std::cout << utils::check_mul<int>(a, b, c, 2, 2, 2) << std::endl;
    std::cout << fabs(-1.34) << std::endl;
    return 0;
}*/


