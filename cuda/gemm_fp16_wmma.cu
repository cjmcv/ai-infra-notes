// %%cuda
/*!
* \brief gemm: C = A * B.
*/
#include <iostream>
#include "time.h"

#include "pocket-ai/engine/cu/common.hpp"
#include "pocket-ai/engine/cu/common_mma.hpp"

using namespace pai::cu;

////////////////////////////////////////////////////////////////////////////////

// Initialize the input data.
void GenHalfMatrix(const int height, const int width, half *mat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = __float2half((float)(rand() % 20 - 10)); // int: -10 ~ 10
            // mat[i*width + j] = __float2half(1); // int: -10 ~ 10
        }
    }
}

// Just for checking the result.
float GetHalfMean(const half* mat, const int height, const int width) {
    int num = height * width;
    float total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += __half2float(mat[i*width + j]);
        }
    }
    return total / num;
}

// Just for checking the result too.
void HalfMatrixPrint(const half* mat, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << __half2float(mat[i*width + j]) << ",";
        }
        std::cout << std::endl;
    }
}

// CPU普通实现版本，主要用于核对后续优化版本结果的正确性
void GemmHostV1(const int M, const int N, const int K,
    const half *A, const int lda,
    const half *B, const int ldb,
    half *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(half) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float acc = 0;
            for (k = 0; k < K; ++k) {
                acc += __half2float(A[i*lda + k])*__half2float(B[k*ldb + j]);
            }
            C[i*ldc + j] = __float2half(acc);
        }
    }
}

// CUDA version 1: 72 ms、
// 基于GemmHostV2直接一一对应改写而成,
// 其中的 bi,bj 使用 blockIdx.x,blockIdx.y 代替
// 其中的 i,j 使用 threadIdx.x,threadIdx.y 代替
// (注意：如GemmHostV2中block应为正方形)
// 所以去掉块内线程i/j和块的bi/bj，只需留下 k 循环.
//
// \ C[ty, tx] = A[ty, k] * B[k, tx]
// for k -> K
//     C[bi*bs + ty, bj*bs + tx] += A[bi*bs + ty, k] * B[k, bj*bs + tx]
__global__ void GemmKernelv1(const int M, const int N, const int K,
                             const half* __restrict__ A, const int lda,
                             const half* __restrict__ B, const int ldb,
                             half* __restrict__ C, const int ldc) {

    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;

    half c_sub_acc = 0;
    for (int k = 0; k < K; k++) {
        c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
    }
    C[gid_y * ldc + gid_x] = c_sub_acc;
}

// // // 一个block有32*32个线程，一个warp 32个线程，则有32个warp。
// // // 一个warp处理16*16个元素，即一个block处理
// // dim3 blocks_per_grid_r(blocks_per_grid.x/16, blocks_per_grid.y/16);
// // GemmWmmaKernelv1<< <blocks_per_grid_r, threads_per_block >> >
// //     (M, N, K, A, lda, B, ldb, C, ldc);   
// __global__ void GemmWmmaKernelv1(const int M, const int N, const int K,
//                                 const half* __restrict__ A, const int lda,
//                                 const half* __restrict__ B, const int ldb,
//                                 half* __restrict__ C, const int ldc) {

//     const int WMMA_M = 16;
//     const int WMMA_N = 16;
//     const int WMMA_K = 16;

//     const int warp_y = blockIdx.y * WMMA_M;
//     const int warp_x = blockIdx.x * WMMA_N;
    
//     if (warp_y >= M && warp_x >= N) {
//         printf("warp_y = %d, warp_x = %d.\n", warp_y, warp_x);
//         return;
//     }

//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> C_frag;
//     wmma::fill_fragment(C_frag, 0.0);

// // #pragma unroll
//     // printf("(w %d, %d), ", warp_x, warp_y);
//     for (int warp_k = 0; warp_k < K; warp_k += WMMA_K) {
//         wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> A_frag;
//         wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> B_frag;

//         wmma::load_matrix_sync(A_frag, A + warp_y * K + warp_k, K);
//         wmma::load_matrix_sync(B_frag, B + warp_k * N + warp_x, N);

//         wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
//     }

//     wmma::store_matrix_sync(C + warp_y * N + warp_x, C_frag, N, wmma::mem_row_major);
// }

// 
template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void GemmWmmaKernelv1(const int M, const int N, const int K,
                                const half* __restrict__ A, const int lda,
                                const half* __restrict__ B, const int ldb,
                                half* __restrict__ C, const int ldc) {
    const int WARP_SIZE = 32;

    const int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int wid_x = gid_x / WARP_SIZE;  // warp维度为(32, 1), 所以x方向要除，y方向不用
    const int wid_y = gid_y;

    // 得到warp id后直接乘以wmma的块大小，就是对应的偏移量，一个warp的线程处理完一个wmma块的计算
    const int fid_x = wid_x * WMMA_N;
    const int fid_y = wid_y * WMMA_M;

    if (fid_y >= M || fid_x >= N) {
        printf("fid_y = %d, fid_x = %d.\n", fid_y, fid_x);
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    for (int fid_K = 0; fid_K < K; fid_K += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + fid_y * K + fid_K, K);
        wmma::load_matrix_sync(B_frag, B + fid_K * N + fid_x, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + fid_y * N + fid_x, C_frag, N, wmma::mem_row_major);
}

float MatrixMulCUDA(int version_id, int step,
                    const int M, const int N, const int K,
                    const half *A, const int lda,
                    const half *B, const int ldb,
                    half *C, const int ldc) {
    GpuTimer gpu_timer;

    const int block_side_size = 32;
    dim3 threads_per_block(block_side_size, block_side_size);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);
    
    // Warm up.
    for (int i=0; i<10; i++) {
        GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    cudaMemset(C, 0, sizeof(half) * M * N);

    // Record the start event
    gpu_timer.Start();

    if (version_id == 0) {
        GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    else if (version_id == 1) {
        // warp维度可认为是 (32，1)
        //   x方向一个block有32个线程合计一个warp (threads_per_block.x / WARP_SIZE)，共处理1*16个元素，
        // 则为一个block处理16个元素需要多少个block，直接用N/16即可。
        //   y方向一个block有32个线程，合计32个warp (threads_per_block.y / 1)，共处理32*16个元素，
        // 则直接用M/(32*16)即可,32*16数值较大，注意向上取整。
        const int WARP_SIZE = 32;
        const int WMMA_M = 16;
        const int WMMA_N = 16;
        const int WMMA_K = 16;
        const int warps_pre_block_x = threads_per_block.x / WARP_SIZE;
        const int warps_pre_block_y = threads_per_block.y / 1;
        dim3 blocks_per_grid_r( DivCeil(N, warps_pre_block_x*WMMA_N), DivCeil(M, warps_pre_block_y*WMMA_M) );
        GemmWmmaKernelv1<WMMA_M, WMMA_N, WMMA_K> << <blocks_per_grid_r, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);
    }
    // Record the stop event
    gpu_timer.Stop();

    return gpu_timer.ElapsedMillis();
}

#define TEST_CUDA_MODULE_UKERNEL(version_id, step)                            \
    do {                                                                      \
        CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice)); \
        CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice)); \
        msec_total = MatrixMulCUDA(version_id, step, height_a, width_b, width_a, d_a, width_a, d_b, width_b, d_c, width_b); \
        CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost)); \
        printf("gpu version %d step %2d -> time: %f s, mean value = %f\n", version_id, step, msec_total/1000.f, GetHalfMean(h_c, height_a, width_b)); \
    } while (0)

int main() {
    int ret = InitEnvironment(0);
    if (ret != 0) {
        printf("Failed to initialize the environment for cuda.");
        return -1;
    }

    // Normal test
    int height_a = 1536, width_a = 4096;
    int height_b = 4096, width_b = 2048;
    // // Test split-k
    // int height_a = 64, width_a = 4096;
    // int height_b = 4096, width_b = 64;
    // // Debug
    // int height_a = 32, width_a = 16;
    // int height_b = 16, width_b = 32;
    if (width_a != height_b) {
        printf("width_a should be equal to height_b.\n");
        return 1;
    }

    const int mem_size_a = sizeof(half) * height_a * width_a;
    const int mem_size_b = sizeof(half) * height_b * width_b;
    const int mem_size_c = sizeof(half) * height_a * width_b;

    half *h_a = (half *)malloc(mem_size_a);
    half *h_b = (half *)malloc(mem_size_b);
    half *h_c = (half *)malloc(mem_size_c);
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        printf("Fail to malloc.\n");
        return 1;
    }

    // Initialize 
    srand(time(NULL));
    GenHalfMatrix(height_a, width_a, h_a);
    GenHalfMatrix(height_b, width_b, h_b);

    // CPU
    // time_t t = clock();
    // GemmHostV1(height_a, width_b, width_a, h_a, width_a,h_b, width_b, h_c, width_b);
    // printf("cpu version 1 -> time: %f s, mean value = %f\n", double(clock() - t)/CLOCKS_PER_SEC, GetHalfMean(h_c, height_a, width_b));
    // HalfMatrixPrint(h_c, height_a, width_b);

    // GPU
    // Allocate memory in host. 
    float msec_total;
    half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
    CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
    CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

    TEST_CUDA_MODULE_UKERNEL(0, 1);
    TEST_CUDA_MODULE_UKERNEL(1, 1);

    // printf("Print output C:\n");
    // for (int i=0; i<height_a; i++) {
    //     for (int j=0; j<width_b; j++) {
    //         printf("%f, ", h_c[i*width_b+j]);
    //     }
    //     printf("\n");
    // }

    // Normal test.
    // GPU Device 0: "Tesla T4" with compute capability 7.5 with 40 multi-processors.
    // cpu version 1 -> time: 168.878989 s, mean value = 1023.253418
    // gpu version 1 step  1 -> time: 0.034384 s, mean value = 1023.097839
    // gpu version 2 step  1 -> time: 0.000495 s, mean value = 1023.098633

    // GPU Device 0: "NVIDIA GeForce RTX 3080" with compute capability 8.6 with 68 multi-processors.

    // gpu version 0 step  1 -> time: 0.013833 s, mean value = 1026.111938
    // gpu version 1 step  1 -> time: 0.001812 s, mean value = 1026.276367
    // gpu version 2 step  1 -> time: 0.000002 s, mean value = 0.000000

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    CleanUpEnvironment();

    return 0;
}
