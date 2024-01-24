I used CUDA GPU programming to implement parallel programming of Strassen Recursive Matrix Multiplication algorithm. I used iterative recursion. An alternative way to implement strassen algorithm in CUDA is dynamic parallelism.

The below explanation of the iterative approach I took:
  Loop thru k'(inclusive to do base case MatMul) to go to deeper levels into the recursion. From Top(root) to bottom(leaf), for each level, I declared two array of matrices; one is called parent matrices, which contain matrices obtained from the upper/parent level, and the other is called child matrices, which will store the output of the current level. At each level except for base case, child matrices are results of sub matrix addition (ex. A11+A22, A21+A22, etc.) which are used as inputs in the subsequent deeper level. At the base case(leaf level), child matrices(output) will be obtained by performing matrix multiplications between an adjacent pair of matrices in the parent matrices; i.e., output1 = parent_matrix1 * parent_matrix2, and output2 = parent_matrix3 * parent_matrix4,... As going to next deeper level, parent<-child, and child<-newly_computed_array.

  From bottom to top, now the output matrix obtained from the base case matrix multiplication has M1,...,M7 for each parent matrix created along the k'-1 recursive calls. Hence, do M1,...,M7 addition and result matrix block assignment for the corresponding parent matrix at each level, and then pop out of the current level to the one upper level. Here, the result matrix from lower level is the input to the upper level.

Discussion:
  While data transferring to GPU device has some cost, the performance on matrix multiplication with matrices with bigger size is very very fast in my implementation compared to the serially executed Strassen at CPU level because my implementation utilizes concurrency at each level by trading off space complexity at GPU as mentioned above. My strassen implementaion outperforms serial execution significantly especially when k' is big and k is big.

Disadvatange:
  1. All of the submatrix addition, matrix multiplication, and M1,...,M7 addition to update the parent matrix processings are handled in the GPU device by parallel execution. However, while achieving fast runtime performance, my implementation comes with some space complexity because O((7/4)^(k') * n^2) memory is needed in GPU to accomodate all child matrices at the base level.

  2. Also, since all submatrix additions for level l in the recursion are performed concurrently in GPU, this may end up creating lots of thread blocks possibly more than the GPU block number limit, which cause to failure of execution; the same issue may arises in matrix multiplications, and matrix additions for the same reason.


---------------------------------------------------------------------------------------------------------------------------------------------------
Program execution:

This is how to compile and run my program.
Compilation: nvcc -ccbin=icc -o filne_name.exe strassen.cu

Execution in terminal: ./file_name.exe is_parellel k' k
where is_parellel, k', and k are integers, and is_parallel = 0 indicates that no parallel computing(is_parallel = 1 indicates GPU parallel computing), and all other int indicating for parallel computing
