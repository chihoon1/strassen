#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>


// max thread dimension for matrix multiplication for kernel function
// warpSize is often 32. And thread Id in x and y axis plays big in matrix multiplication computation
// if MAX_THR_DIM_MUL > sqrt(warpSize), threads executed regarldess of their Id order
// This can cause MatMul_kernel function not working correctly because A_sub and B_sub filled incorrectly
#define MAX_THR_DIM_MUL 4

int BLOCK_SIZE;
int EPS = 0; // 0.01;
int DEBUG = 0;

typedef struct {
    // 1D array representation of 2D matrix (row-major)
    int width;  // dimension of a row space
    int height;  // dimension of a column space
    int stride;  // stride for matrix iteration. useful for submatrix as elem is a ptr
    int *elem;  // ptr to the first element in the 1D representation of matrix
} Matrix;

void std_MatMul(const Matrix A, const Matrix B, Matrix *C) {
    int m, n, step;
    m = C->height; n = C->width; step = C->stride;
    for (int i = 0; i < m; i++) {  // row iteration
        for (int j = 0; j < n; j++) {  // column iteration
            for (int k = 0; k < A.width; k++) {  // inner product iteration
                // A.width == B.height for mat mul to be defined
                // C[i,j] = <A[i,:], B[:,j]> == C[i,j] += A[i,k] * B[k,j]
                C->elem[i*step + j] = C->elem[i*step + j] + A.elem[i*A.stride + k] * B.elem[k*B.stride + j];
            }
        }
    }
}

Matrix std_MatAdd(const Matrix A, const Matrix B, int is_subtraction) {
    if (is_subtraction != -1) {
        // is_subtraction == 1 means matrix addition. -1 means matrix subtraction (A-B)
        is_subtraction = 1;
    }
    Matrix C;
    C.height = A.height; C.width = A.width; C.stride = A.width;
    size_t size = C.height * C.width * sizeof(int);
    C.elem = (int*) malloc(size);
    for (int i = 0; i < C.height; i++) {  // row iteration
        for (int j = 0; j < C.width; j++) {  // column iteration
            C.elem[i*C.stride + j] = A.elem[i*A.stride + j] + is_subtraction *  B.elem[i*B.stride + j];
            
        }
    }
    return C;
}

void print_matrix(const Matrix *mat) {
    printf("Print Matrix\n");
    for (int i = 0; i < mat->width; i++) {
        printf("%d:\t", i);
        for (int j = 0; j < mat->height; j++) {
            printf("%d ", mat->elem[i*mat->stride + j]);
        }
        printf("\n");
    }
}

Matrix get_submatrix(const Matrix A, int row_idx, int col_idx, int sub_n) {
    // param: sub_n is the row or column dimension of the input matrix A
    // assume square matrix
    Matrix Asub;
    Asub.height = sub_n; Asub.width = sub_n; Asub.stride = A.stride;
    Asub.elem = &A.elem[row_idx * A.stride + col_idx];
    return Asub;
}

void partial_MatCpy(const Matrix b, Matrix *d_C, int row, int col) {
    // copy elements of matrix b into the block of matrix d_C, starting at the given row and col
    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            d_C->elem[ (row + i) * d_C->stride + col+j] = b.elem[i * b.stride + j];
        }
    }
}

void init_intMat(int sub_n, Matrix *M) {
    // param: sub_n is the row or column dimension of the input matrix A
    // initialize square matrix M with given dimension by memory allocation
    size_t size = sub_n * sub_n * sizeof(int);
    M->stride = sub_n;
    M->height = sub_n; M->width = sub_n;
    M->elem = (int*) malloc(size);
    for (int i = 0; i < sub_n * sub_n; i++) {
        M->elem[i] = 0;  // initialize to the zero matrix
    }
}


void seq_strassen_recursion(int level, int k_prime, const Matrix d_A, const Matrix d_B, Matrix *d_C) {
    int n, num_subs;
    n = d_C->width;  // This function assumes square matrix
    num_subs = 2;  // num of row or col in submatrix(square matrix assumed)
    int sub_n = n / num_subs;
    if (level > k_prime || n <= 1) {
        std_MatMul(d_A, d_B, d_C);
        // printf("Base Seq\n"); print_matrix(d_C);  // debugging purpose
        return;       
    }
    // obtain submatrices
    Matrix A11, A12, A21, A22, B11, B12, B21, B22;
    A11 = get_submatrix(d_A, 0, 0, sub_n);
    A21 = get_submatrix(d_A, sub_n, 0, sub_n);
    A12 = get_submatrix(d_A, 0, sub_n, sub_n);
    A22 = get_submatrix(d_A, sub_n, sub_n, sub_n);
    B11 = get_submatrix(d_B, 0, 0, sub_n);
    B21 = get_submatrix(d_B, sub_n, 0, sub_n);
    B12 = get_submatrix(d_B, 0, sub_n, sub_n);
    B22 = get_submatrix(d_B, sub_n, sub_n, sub_n);
    // printf("level %d B11 sub_n %d\n", level, sub_n); print_matrix(&B11); // debugging purpose
    // create variables to store matrix Mi
    Matrix M1, M2, M3, M4, M5, M6, M7;
    init_intMat(sub_n, &M1); init_intMat(sub_n, &M2);
    init_intMat(sub_n, &M3); init_intMat(sub_n, &M4);
    init_intMat(sub_n, &M5); init_intMat(sub_n, &M6);
    init_intMat(sub_n, &M7);
    
    // recursively compute mat mul M1, M2, M3, M4, M5, M6, and M7
    // compute the sum of the two sub-matrices
    Matrix S1, S2, S3, S4, S5, S6, S7, S8, S9, S10;
    S1 = std_MatAdd(A11, A22, 1); S2= std_MatAdd(B11, B22, 1);
    S3 = std_MatAdd(A21, A22, 1); S4 = std_MatAdd(B12, B22, -1); S5 = std_MatAdd(B21, B11, -1);
    S6 = std_MatAdd(A11, A12, 1);
    S7 = std_MatAdd(A21, A11, -1); S8 = std_MatAdd(B11, B12, 1);
    S9 = std_MatAdd(A12, A22, -1); S10 = std_MatAdd(B21, B22, 1);
    // printf("Si\n"); print_matrix(&S1); printf("Sj\n"); print_matrix(&S2);  // debugging
    // printf("Mi\n"); print_matrix(&M1);  // debugging
    seq_strassen_recursion(level+1, k_prime, S1, S2, &M1);
    seq_strassen_recursion(level+1, k_prime, S3, B11, &M2);
    seq_strassen_recursion(level+1, k_prime, A11, S4, &M3);
    seq_strassen_recursion(level+1, k_prime, A22, S5, &M4);
    seq_strassen_recursion(level+1, k_prime, S6, B22, &M5);
    seq_strassen_recursion(level+1, k_prime, S7, S8, &M6);
    seq_strassen_recursion(level+1, k_prime, S9, S10, &M7);
    
    // compute four blocks of matrix by adding Mis according to strassen algorithm
    Matrix c1, c2, c3, c4;
    c1 = std_MatAdd(std_MatAdd(M1, M4, 1), std_MatAdd(M7, M5, -1), 1);
    c2 = std_MatAdd(M3, M5, 1);
    c3 = std_MatAdd(M2, M4, 1);
    c4 = std_MatAdd(std_MatAdd(M1, M2, -1), std_MatAdd(M3, M6, 1), 1);
    
    // copy the block matrices to the result matrix in block order
    partial_MatCpy(c1, d_C, 0, 0);
    partial_MatCpy(c2, d_C, 0, sub_n);
    partial_MatCpy(c3, d_C, sub_n, 0);
    partial_MatCpy(c4, d_C, sub_n, sub_n);
}


int is_sameMat(const Matrix A, const Matrix B) {
    // check each element of A and B to see if they are identical matrices
    // elem of A and elem of B in the corresponding position must be differ <= eps
    // params A and B must be the same size matrices
    // globabl variable EPS is the tolerable value different between elements (due to rounding error in data processing)
    // If identical, return 0. Otherwise, return number of mismatched elements between A and B
    int count = 0;
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            if ((A.elem[i * A.stride + j] - B.elem[i * B.stride + j]) > EPS) {
                count++;
            }
        }
    }
    return count;
}


/*
------------------------------------------------------------------
CUDA parallel computing functions
From Below to the main function, all codes are for GPU programming
------------------------------------------------------------------
*/
__global__ void get_warpSize(int *warp_size) {
    *warp_size = warpSize;
}

__global__ void init_zero_level_parent(int *parent_mats, const int* A, const int* B, int n) {
    // Initialize parent matrix at the initial level
    // every even matrix is A and every odd matrix is B for simlplicity of submat_add kernel execution
    // Copy A or B to one destination matrix to store a copy. And one thread copy one elem
    // first n*n elem in the parent_mats array will be designated for one matrix, second n*n elem for another matrix, and etc.
    float frac = ((float) (n * n))/ (float) blockDim.x;  // Here, blockDim.x is the num_threads
    int blocks_per_mat = (int) ceil(frac);
    int mat_loc = blockIdx.x / blocks_per_mat;  // if 0, matrix A. Else, matrix B
    // this block's first thread's position in an input matrix A or B elem array
    int block_pos_in_mat = blockIdx.x % blocks_per_mat;
    int moving_idx = threadIdx.x + block_pos_in_mat * blockDim.x;
    if (mat_loc == 1) {  // copy B
        parent_mats[(mat_loc * n * n) + moving_idx] = B[moving_idx];
    } else {  // even, so copy A
        parent_mats[(mat_loc * n * n) + moving_idx] = A[moving_idx];
    }
}

__global__ void print_matrix_device(int *mats, int matrix_pos, int stride) {
    // print matrix in device. One matrix per block
    printf("Matrix %d printing\n", matrix_pos);
    int mat_size = stride * stride;
    int start_idx = matrix_pos * mat_size;
    for (int i = 0; i < mat_size; i++) {
        if (i % stride == 0) {
            int group_id = i / stride;
            printf("\n%d: ", group_id);
        }
        printf("%d\t", mats[start_idx + i]);
        // printf("%d\t", start_idx + i);  // debugging purpose
    }
    printf("\n");
}

__global__ void submat_add(int *parent_mats, int *child_mats, int sub_n) {
    // Expect a 1d block and 1d grid
    // One thread will perform addition on one element of the matrix
    // Expect number of threads == number of elements in a submatrix
    // Total 14 types of matrix summation in the following order:
    // A11+A22, B11+B22, A21+A22, B11, A11, B12-B22, A22, B21-B11, A11+A12, B22, A21-A11, B11+B12, A12-A22, B21+B22
    // a matrix elem in parent_mats and child_mats have dimension of sub_n*2 by sub_n*2
    int n = sub_n * 2;  // dimension of a matrix in parent_mats
    float frac = ((float) (sub_n * sub_n))/ (float) blockDim.x;  // Here, blockDim.x is the num_threads    
    int blocks_per_mat = (int) ceil(frac);
    // location in child_mats where output matrix will be stored
    int output_label = blockIdx.x / blocks_per_mat;
    int output_loc = (output_label * sub_n * sub_n);
    int submat_type = output_label % 14;  // indicates which type of submatrix summation among 14 types
    // location of matrix in parent_mat where an input of this function is stored
    int input_loc = (2 * (output_label / 14) + output_label % 2) * n * n;  // output_loc % 2 is even, then A. Else, B.
    int parent_row_idx, parent_col_idx;
    int block_pos_in_mat = blockIdx.x % blocks_per_mat; // this block's position in a output matrix sum
    parent_row_idx = (threadIdx.x + block_pos_in_mat * blockDim.x) / sub_n;
    parent_col_idx = (threadIdx.x + block_pos_in_mat * blockDim.x) % sub_n;
    int row_1, col_1, row_2, col_2, parent1_idx, parent2_idx;
    // No Warp Divergent as all the threads in the same block compute the same submat_type summation
    int addition_type = 1;  // 1 is addition, -1 is subtraction, and 0 is a copy of a submatrix
    // A11+A22, B11+B22, A21+A22, B11, A11, B12-B22, A22, B21-B11, A11+A12, B22, A21-A11, B11+B12, A12-A22, B21+B22
    if (submat_type == 0 || submat_type == 1) {  // A11+A22 or B11+B22
        row_1 = 0; col_1 = 0; row_2 = sub_n; col_2 = sub_n;
    } else if (submat_type == 2 || submat_type == 13) { // A21+A22 or B21+B22
        row_1 = sub_n; col_1 = 0; row_2 = sub_n; col_2 = sub_n;
    } else if (submat_type == 3 || submat_type == 4){  // B11 or A11
        row_1 = 0; col_1 = 0; row_2 = 0; col_2 = 0;
        addition_type = 0;
    } else if (submat_type == 5 || submat_type == 12) {  // B12-B22 or A12-A22
        row_1 = 0; col_1 = sub_n; row_2 = sub_n; col_2 = sub_n;
        addition_type = -1;
    } else if (submat_type == 6 || submat_type == 9) { // A22 or B22
        row_1 = sub_n; col_1 = sub_n; row_2 = sub_n; col_2 = sub_n;
        addition_type = 0;
    } else if (submat_type == 7 || submat_type == 10) {  // B21-B11 or A21-A11
        row_1 = sub_n; col_1 = 0; row_2 = 0; col_2 = 0;
        addition_type = -1;
    } else {  // A11+A12 or B11+B12
        row_1 = 0; col_1 = 0; row_2 = 0; col_2 = sub_n;
    }
    parent1_idx = (row_1 + parent_row_idx) * n + (col_1 + parent_col_idx);  // index to input matrix1
    parent2_idx = (row_2 + parent_row_idx) * n + (col_2 + parent_col_idx);  // index to input matrix2
    int child_idx = block_pos_in_mat * blockDim.x + threadIdx.x;
    if (child_idx < (sub_n*sub_n)) {
        // To make sure not write the result out of the output matrix element's index
        int p1_idx, p2_idx, c_idx; c_idx = output_loc + child_idx;
        p1_idx = input_loc + parent1_idx; p2_idx = input_loc + parent2_idx;
        child_mats[c_idx] = parent_mats[p1_idx] + addition_type * parent_mats[p2_idx];
    }
}

__device__ int get_elem(int *mats, int idx) {
    return mats[idx];
}


__global__ void MatMul_kernel(int *input_mats, int *output_mats, int n) {
    // Expects square 2d thread blocks and 1d grid
    // Input matrices expect to be square matrices
    int num_threads_block = blockDim.x * blockDim.y;  // number of threads in thread block
    float frac = ((float)(n * n)) / ((float) num_threads_block);
    int blocks_per_mat = (int) ceil(frac);  // number of blocks needed to do mat mul per output matrix
    int block_label = blockIdx.x % blocks_per_mat;
    // num_blocks per axis in an ouput matrix(same as input matrix as square matrix expected)
    int blocks_per_axis = (int) ceil((float) n / (float) blockDim.x);
    int block_row = block_label / blocks_per_axis; int block_col = block_label % blocks_per_axis;
    // label(id) and location of output matrix in output_mats
    int output_label = blockIdx.x / blocks_per_mat; int output_loc = output_label * n *n;
    int pair_idx = 2 * output_label;  // starting index of a input matrix pair
    int input_loc1 = pair_idx * n * n; int input_loc2 = (pair_idx + 1) * n * n;
    int thr_row = threadIdx.x; int thr_col = threadIdx.y;  // thread's row and col index
    int inner_prod = 0;
    // utilizing shared variables of submatrices from input matrices to reduce memory traffic between SM and device mem
    __shared__ int A_sub[MAX_THR_DIM_MUL][MAX_THR_DIM_MUL], B_sub[MAX_THR_DIM_MUL][MAX_THR_DIM_MUL]; 
    for (int i = 0; i < blocks_per_axis; i++) {
        // reminder: n is a stride(width) of a 1d array representation of row-major matrix
        // Let current block of threads called T_i,j where i(row) and j(col) are y and x axis, resepctively
        // Get submats of A for current block in order of: A_i,1, A_i,2, A_i,3,..., A_i,k
        // Get submats of B for current block in order of: B_1,j, A_2,j, A_3,j,..., A_k,j
        // Then, C_i,j, the square region in output matrix C, equal to A_i,1*B_1j + ... + A_i,k * A_k,j
        // block_row and block_col represent i and j in C_i,j
        int A_idx = input_loc1 + (thr_row + block_row*blockDim.y)*n  + (thr_col + i*blockDim.x);
        int B_idx = input_loc2 + (thr_row + i*blockDim.y)*n + (thr_col + block_col*blockDim.x);
        // each thread fill out one elem in A_sub and B_sub
        A_sub[thr_row][thr_col] = get_elem(input_mats, A_idx);
        B_sub[thr_row][thr_col] = get_elem(input_mats, B_idx);
        __syncthreads();  // need to synchornize here so that the submatrix is filled
        // compute the inner product of a row in A_sub and a col in B_sub
        for (int l=0; l < blockDim.x; l++) {  // square block, so x and y axis dimension same
            inner_prod = inner_prod + A_sub[thr_row][l] * B_sub[l][thr_col];
        }
    }
    // similar indexing approach used in A_idx and B_idx but replaced i with blockDim.x and y
    // because different thread block computes Mat Mul in different square region(block) in the output matrix
    int output_idx = output_loc + (block_row*blockDim.y + thr_row)*n + (block_col*blockDim.x + thr_col);
    output_mats[output_idx] = inner_prod;
}

__global__ void Mi_add_forBlock_Kernel(int *input_mats, int *output_mats, int sub_n) {
    // Expect a 1d block and 1d grid
    // intput_mats and output_mats store in the order of M1,...,M7
    int n = sub_n * 2;  // dimension of a matrix in output_mats
    int num_elem_input = sub_n * sub_n;
    float frac = ((float) (num_elem_input))/ (float) blockDim.x;  // Here, blockDim.x is the num_threads    
    int blocks_per_mat = (int) ceil(frac);  // number of blocks of threads needed for one submatrix(region) in the output matrix
    int output_label = blockIdx.x / (4*blocks_per_mat);  // 4==4 quadrants in the output matrix
    int output_loc = (output_label * n * n);  // location of output matrix of this block in the output_mats
    // input group's location (==index of first element of first matrix(=M1) in input_mats). one group==M1,...,M7
    int input_loc = (7 * output_label) * num_elem_input;  // 7==M1,...,M7
    int quadrant_type = (blockIdx.x / blocks_per_mat) % 4;  // indicates which type of quadrant in a output matrix that this block updates
    int block_pos_in_mat = blockIdx.x % blocks_per_mat;  // this block's position in a matrix in input_mats
    // block_pos_in_mat * blockDim.x == total number of threads exist upto the previous block id in input matrix this block uses
    int input_idx = block_pos_in_mat * blockDim.x + threadIdx.x;  // index in an input matrix (1D reprensentation)
    int output_row_idx, output_col_idx, output_idx;
    // iteration for row and col follow the submatrix's row and col because a matrix filled out as a block in quadrant
    output_row_idx = (input_idx) / sub_n;
    output_col_idx = (input_idx) % sub_n;
    // quadrant order: C11, C12, C21, C22
    int M1_pos, m2_idx, m3_idx, m4_idx, m5_idx, m6_idx, m7_idx;
    M1_pos = input_loc + input_idx;  // current thread's position at M1 in this group in input_mats
    if (input_idx < (num_elem_input)) {
        // To make sure not write the result out of the output matrix element's index
        switch (quadrant_type) {
            case 0:  // C11
                output_idx = output_loc + (0 + output_row_idx) * n + (0 + output_col_idx);
                m4_idx = M1_pos + 3*num_elem_input; m5_idx = M1_pos + 4*num_elem_input;
                m7_idx = M1_pos + 6*num_elem_input;
                output_mats[output_idx] = input_mats[M1_pos] + input_mats[m4_idx] - input_mats[m5_idx] + input_mats[m7_idx];
                break;
            case 1:  // C12
                output_idx = output_loc + (0 + output_row_idx) * n + (sub_n + output_col_idx);
                m3_idx = M1_pos + 2*num_elem_input; m5_idx = M1_pos + 4*num_elem_input;
                output_mats[output_idx] = input_mats[m3_idx] + input_mats[m5_idx];
                    break;
            case 2:  // C21
                output_idx = output_loc + (sub_n + output_row_idx) * n + (0 + output_col_idx);
                m2_idx = M1_pos + num_elem_input; m4_idx = M1_pos + 3*num_elem_input;
                output_mats[output_idx] = input_mats[m2_idx] + input_mats[m4_idx];
                break;
            default:  // C22
                output_idx = output_loc + (sub_n + output_row_idx) * n + (sub_n + output_col_idx);
                m2_idx = M1_pos + num_elem_input; m3_idx = M1_pos + 2*num_elem_input;
                m6_idx = M1_pos + 5*num_elem_input;
                output_mats[output_idx] = input_mats[M1_pos] - input_mats[m2_idx] + input_mats[m3_idx] + input_mats[m6_idx];
                break;
        }
    }
}


/*
--------------------------
        MAIN FUNCTION
--------------------------
*/
// Input:
//  is_parallel is int and if the value == 0 indicates that no parallel computing, and all others int indicating for parallel computing
//  k = log2(n) where matrices to be multiplied are size n * n
//  k' = lowest level in recursion. k' > 1
// Compile the codes:
//      nvcc -ccbin=icc -o filne_name.exe strassen.cu
// Run in terminal
//      ./file_name.exe is_parellel k' k
// where is_parellel, k', and k are integers
// is_parallel == 0 indicates that no parallel computing, and all other int indicating for parallel computing
int main(int argc, char *argv[]) {
    int k, k_prime, is_parallel; float time;
    if (argc != 4) {
        printf("REQUIRES 3 INPUTS FOR PROGRAM EXECUTION\n");
        printf("./file_name.exe is_parellel k' k\n");
        printf("where is_parallel == 0 indicates that no parallel computing,\
                and all other int indicating for parallel computing\n");
        exit(1);
    }
    k = atoi(argv[argc-1]);
    k_prime = atoi(argv[argc-2]);
    if (k_prime > k) {
        // k' can't be > k becuase input matrix is of size n*n where n=2^k
        // and a matrix is of size n/(2^k')*n/(2^k') at level k'
        // when k'>k, that size is 0*0, which is not allowed for a matrix
        k_prime = k;
    }
    if (atoi(argv[argc-3]) == 0) {  // Serial computing for strassen matrix multiplication
        is_parallel = 0;
    } else {  // Parallel computing for strassen matrix multiplication
        is_parallel = 1;
    }
    // printf("k: %d\n", k); printf("k': %d\n", k_prime);
    int n = pow(2, k);
    // printf("n: %d\n", n); printf("Is Parallel Computing? %d\n", is_parallel);
    // Input Matrices and output matrix set up and initialization
    Matrix h_A, h_B;
    h_A.stride = h_A.width = n; h_A.height = n;
    h_A.elem = (int *) malloc(h_A.width*h_A.height*sizeof(int));
    h_B.stride = h_B.width = n; h_B.height = n;
    h_B.elem = (int *) malloc(h_B.width*h_B.height*sizeof(int));
    Matrix h_C;
    h_C.stride = h_C.width = h_B.width; h_C.height = h_A.height;
    h_C.elem = (int *) malloc(h_C.width*h_C.height*sizeof(int));
    Matrix seq_h_C;  // the result matrix to test correctness of strassen matrix multiplication
    seq_h_C.stride = seq_h_C.width = h_B.width; seq_h_C.height = h_A.height;
    seq_h_C.elem = (int *) malloc(seq_h_C.width*seq_h_C.height*sizeof(int));
    float rmax, rmin; rmax = 10; rmin=-10;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A.elem[i*h_A.width + j] = ( (float) rand() / RAND_MAX) * (rmax-rmin)+rmin;
            h_B.elem[i*h_B.width + j] = ( (float) rand() / RAND_MAX) * (rmax-rmin)+rmin;
            h_C.elem[i*h_C.width + j] = (int) 0;
            seq_h_C.elem[i*h_C.width + j] = (int) 0;
        }
    }
    if (is_parallel == 1) {
        printf("Parallel Compute\n");
    // CUDA programming
    // Not timed with Cuda function because recursion done at cpu level and every iteration is synchronized
    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);
    // set the block size according to the warp size
    int *warp_size;
    cudaMalloc(&warp_size, (size_t) sizeof(int));
    get_warpSize<<<1,1>>>(warp_size);
    cudaMemcpy(&BLOCK_SIZE, warp_size, (size_t) sizeof(int), cudaMemcpyDeviceToHost);
    // data transfer to GPU device
    size_t mat_size = n*n*sizeof(int);
    int *d_A, *d_B;
    cudaMalloc(&d_A, mat_size); cudaMemcpy(d_A, h_A.elem, mat_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, mat_size); cudaMemcpy(d_B, h_B.elem, mat_size, cudaMemcpyHostToDevice);
    int *Mats_parent;  // contain input matrices for current level of recursion
    // initialize Mats_parent for the first recursion level(root)
    cudaMalloc(&Mats_parent, 2*mat_size);
    int number_threads = min(n*n, BLOCK_SIZE);
    int blocks_per_mat = (int) ceil(((float) n * n)/((float) number_threads));
    init_zero_level_parent<<<2*blocks_per_mat, number_threads>>>(Mats_parent, d_A, d_B, n);
    cudaDeviceSynchronize();
    /*
    Recursion Implementation Explanation:
        loop thru k_prime(inclusive to do base case MatMul), then going down into the recursion
        parent(upper level) matrices and child matrices(lowever level that requires parent matrices as its input)
        as level goes up, parent <- child, and child <- newly computed
        child matrices are results of subMat_AddKernel(parellel for all child) or MatMul_kernel(if base case).
        
        After base case index,
        do Mi addition(not submat) and result matrix block assignment by popping out of recursion one by one level
        the result matrix from lower level is now the input to the upper level
    I used iterative recursion because nested kernel call not allowed with the compiler in the machine I use
    ONE CAVEAT: This algorithm implementation requires intensive number of thread blocks at deeper level
                However, GPU has limit on number of blocks
    */
    // From Top to Down in recursion
    int sub_n = n / 2; int base_level_n;  // dimension of matrix at the base case level
    // num_arr is equivalent to number of matrices called for strassen mat mul at current level
    // num_mats is number of output matrices after computation (either summation or multiplication if base case)
    int num_mats, num_arr;
    for (int level = 0; level <= k_prime; level++) {
        // Initialize matrices that need to compute all M1,...,M7 at current recursion level
        int *Mats_child;  // an array containing num_arr * num_mat number of submatrices of size sub_n*sub_n
        size_t submats_size;
        if (level == k_prime || sub_n < 1) {  // at k_prime level, or submatrix is smallest possible
            // recursion base case Matrix Multiplication
            num_mats = 7; num_arr = pow(7, level-1);
            // sub_n < 1, then the matrices in Mat_parents are 1*1 matrix(scalar)
            if (sub_n < 1) { base_level_n = 1; }
            else { base_level_n = sub_n = sub_n * 2; }  // otherwise, both same dimension as matrix in Mat_parents
            submats_size = num_arr * num_mats * base_level_n * base_level_n * sizeof(int);
            cudaMalloc(&Mats_child, submats_size);
            int thr_dim = min(base_level_n, MAX_THR_DIM_MUL); // At most, 4*4 2d thread array per block
            dim3 dimBlock(thr_dim, thr_dim);
            int grid_x = (int) ceil(((float) base_level_n * base_level_n)/((float) thr_dim * thr_dim));
            MatMul_kernel<<<grid_x * num_arr * num_mats, dimBlock>>>(Mats_parent, Mats_child, base_level_n);
            level = k_prime + 1;  // This will ensure break out of the loop if sub_n == 1
            cudaDeviceSynchronize();
        } else {  // recursive steps
            // k_prime - 1 level is the leaf level (the base case level)
            // However, will do the matrix multiplication at k_prime level
            // And k_prime - 1 level will compute matrix sums needed for matrix multiplication at base case mat mul
            num_mats = 14; num_arr = pow(7, level);
            submats_size = num_arr * num_mats * sub_n * sub_n * sizeof(int);
            cudaMalloc(&Mats_child, submats_size);
            // Total 14 types of matrix summation in the following order:
            // A11+A22, B11+B22, A21+A22, B11, A11, B12-B22, A22, B21-B11, A11+A12, B22, A21-A11, B11+B12, A12-A22, B21+B22
            int num_threads = min(sub_n*sub_n, BLOCK_SIZE);  // 512 == arbitrary per block thread limits
            int grid_x = (int) ceil(((float) sub_n * sub_n)/((float) num_threads));
            submat_add<<<grid_x * num_arr * num_mats, num_threads>>>(Mats_parent, Mats_child, sub_n);
            cudaDeviceSynchronize();
        }
        // Free the memory allocated for parents matrices in GPU if level >= 1
        // Parent Matrices elements are not used in further recursive computation
        cudaFree(Mats_parent);
        Mats_parent = Mats_child;  // current child will be parent in next level
        /*
        // Device Memory usage checking
        // This strassen implementation has not light space complexity
        // Space Complexity at level l in recursion is O((7/4)^l * n^2)
        // where l is the current level in recursion and n^2 is the total elem in input matrix
        // At base case, O((7/4)^(k') * n^2).
        // So, the program may fail if the spcace complexity is close to the device memory
        size_t free_byte ;
        size_t total_byte ;
        cudaMemGetInfo( &free_byte, &total_byte );
        cudaDeviceSynchronize();
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",\
                used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        */
        sub_n = sub_n / 2;
    }
    // From Down to Top for popping out of recursion
    sub_n = base_level_n;
    int *input_Mats; input_Mats = Mats_parent;  // insignificant. Just using different variable name for readability
    for (int level = k_prime - 1; level >= 0 ; level--) {
        int tot_num_mats = pow(7, level);  // 7 matrices: M1,...,M7. And 7*level-1 of Ms at curr level
        int *output_Mats;  // array of matrices contating matrix M1,...,M7 at current level
        int n = sub_n * 2;
        size_t output_mats_size = tot_num_mats * n * n * sizeof(int);
        cudaMalloc(&output_Mats, output_mats_size);
        int num_threads = min(sub_n*sub_n, BLOCK_SIZE);
        int grid_x = (int) ceil(((float) sub_n * sub_n)/((float) num_threads));
        int grid_dimension = grid_x * tot_num_mats * 4;  // 4 == four quadrant in an output matrix
        Mi_add_forBlock_Kernel<<<grid_dimension, num_threads>>>(input_Mats, output_Mats, sub_n);
        cudaDeviceSynchronize();
        cudaFree(input_Mats);
        input_Mats = output_Mats;
        sub_n = n;
    }  // Won't exit this loop until all matrix multiplication steps completed due to DeviceSynchorize
    // At this point, input_Mats contains the output matrix of Matrix Multiplication
    cudaMemcpy(h_C.elem, input_Mats, (size_t) h_C.height * h_C.width * sizeof(int), cudaMemcpyDeviceToHost);
    
    // destruct cuda variables in the device
    cudaFree(input_Mats);
    cudaFree(warp_size);
    cudaFree(d_A);
    cudaFree(d_B);
    // compute the GPU exeuction time
    clock_gettime(CLOCK_REALTIME, &stop);
    time = (stop.tv_sec-start.tv_sec)+0.000000001*(stop.tv_nsec-start.tv_nsec);
    }  // CUDA parallel computing if statement ends here
    else {  // Serial Computing (is_parallel == False)
        // p=1 computation and timing
        printf("Serial Compute\n");
        // compute the CPU serial exeuction time
        struct timespec seq_start, seq_stop;
        clock_gettime(CLOCK_REALTIME, &seq_start);
        seq_strassen_recursion(1, k_prime, h_A, h_B, &h_C);
        clock_gettime(CLOCK_REALTIME, &seq_stop);
        time = (seq_stop.tv_sec-seq_start.tv_sec)+0.000000001*(seq_stop.tv_nsec-seq_start.tv_nsec);
    }
    // standard matrix multiplication compuation for comparison to check whether strassen algorithm computed correctly
    std_MatMul(h_A, h_B, &seq_h_C);
    
    int error = is_sameMat(h_C, seq_h_C);  // number of mismatches if strassen computed mat mul has an error

    // print time taken
    printf("K: %d, Matrix Shape: %d * %d, K': %d, Parallel Computing? %d, time (sec) = %8.4f, error? %d\n",\
        k, n, n, k_prime, is_parallel, time, error);
    int correct; if (error > 0) correct = 0; else correct = 1;
    printf("Computed Correctly? %d\n", correct);
    if (DEBUG > 0) {  // global variable DEBUG set true so test the correctness of algorithm
        // print input and output if DEBUG >= 1
        printf("Input Matrix A\n");
        print_matrix(&h_A);
        printf("Input Matrix B\n");
        print_matrix(&h_B);
        printf("Output Matrix C by Strassen\n");
        print_matrix(&h_C);
        if (DEBUG > 1) {
            // print stand matrix multiplication output if DEBUG >= 2
            printf("Output Matrix C by standard O(n^3) Mat Mul\n");
            print_matrix(&seq_h_C);    
        }
    }
    printf("\n");
}
