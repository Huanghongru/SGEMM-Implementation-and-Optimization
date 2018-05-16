#include <iostream>
#include <cstdlib>

template <typename T>
T** random_fill_matrix(int row, int col, int min=0, int max=100) {
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

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            mat[i][j] = std::rand() % (max-min+1) + min;
        }
    }
    return mat;
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
    

void hello() {
    std::cout << "hello" << std::endl;
}

