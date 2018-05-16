#include "utils.cpp"

int main() {
    int row = 15, col = 15;
    int **a = random_fill_matrix<int>(row, col, 0, 10);
    print_mat(a, row, col);
    return 0;
}

