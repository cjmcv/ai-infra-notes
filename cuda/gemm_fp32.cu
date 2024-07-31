%%cuda
/*!
* \brief gemm: C = A * B.
*/
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "time.h"

////////////////
// Macro.
////////////////
#define CUDA_CHECK(condition) \
    do { \
        cudaError_t error = condition; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA_CHECK error in line %d of file %s : %s \n", \
                    __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
            exit(EXIT_FAILURE); \
        } \
    } while(0);

////////////////
// Structure.
////////////////

// Timer for cuda.
struct GpuTimer {
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void Start() {
        cudaEventRecord(start_, NULL);
    }
    void Stop() {
        cudaEventRecord(stop_, NULL);
    }
    float ElapsedMillis() {
        float elapsed;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }

    cudaEvent_t start_;
    cudaEvent_t stop_;
};

////////////////
// Function.
////////////////

// 
int InitEnvironment(const int dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        return 1;
    }
    fprintf(stderr, "GPU Device %d: \"%s\" with compute capability %d.%d with %d multi-processors.\n\n", 
      dev_id, device_prop.name, device_prop.major, device_prop.minor, device_prop.multiProcessorCount);

    return 0;
}

void CleanUpEnvironment() {
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CUDA_CHECK(cudaDeviceReset());
}

////////////////////////////////////////////////////////////////////////////////

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
        }
    }
}

// Just for checking the result.
float GetMean(const float* mat, const int height, const int width) {
    int num = height * width;
    float total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += mat[i*width + j];
        }
    }
    return total / num;
}

// Just for checking the result too.
void MatrixPrint(const float* mat, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << mat[i*width + j] << ",";
        }
        std::cout << std::endl;
    }
}

// CPU version 1: 1583 ms
// 普通实现版本
void GemmHostV1(const int M, const int N, const int K,
    const float *A, const int lda,
    const float *B, const int ldb,
    float *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < K; ++k) {
                C[i*ldc + j] += A[i*lda + k]*B[k*ldb + j];
            }
        }
    }
}

// CPU version 2: 3389 ms
// 按i和j方向分块的矩阵乘法，便于改写成cuda
// （暂时省略边界处理）
void GemmHostV2(const int M, const int N, const int K,
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {
    int bi, bj;
    int i, j, k;
    const int block_size = 32;
    int block_num_M = M / block_size;
    int block_num_N = N / block_size;
    memset(C, 0, sizeof(float) * ldc * M);

    // Loop over all of the blocks.
    for (bi = 0; bi < block_num_M; ++bi) {
        for (bj = 0; bj < block_num_N; ++bj) {
            // Loop over all of the elements in a block.
            for (i = bi*block_size; i < (bi + 1)*block_size; ++i) {
                for (j = bj*block_size; j < (bj + 1)*block_size; ++j) { 
                    for (k = 0; k < K; ++k) {
                        C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
                    }
                }
            }
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
                             const float *A, const int lda,
                             const float *B, const int ldb,
                             float *C, const int ldc) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float c_sub_acc = 0;
    for (int k = 0; k < K; k++) {
        c_sub_acc += A[i * lda + k] * B[k * ldb + j];
    }
    C[i * ldc + j] = c_sub_acc;
}

// CUDA version 2.
// 使用共享内存优化：先将数据从全局内存拷贝到共享内存，在共享内存中进行乘加运算，最后写回全局内存
//    因为共享内存以block划分，所以需要将逐个block的数据填充到shared[threadIdx.y][threadIdx.x]中，
// 则A和B矩阵均往各自K方向取block的数据进行填充。所以k方向多拆一个循环来索引块。
// 最终从多次读取全局内存计算 变成 一次读取全局内存到共享内存，多次读取共享内存计算
// 参考host端三层循环，对于最内层循环，A读取会重复 j 次，B读取会重复 i 次
// ps: 用template <int BLOCK_SIZE>的原因是kernel内以固定大小的方式开辟共享内存空间，无法使用变量blockDim
template <int BLOCK_SIZE>
__global__ void GemmKernelv2(const int M, const int N, const int K,
                             const float *A, const int lda,
                             const float *B, const int ldb,
                             float *C, const int ldc) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    float c_sub_acc = 0;
    // 按 K 方向分块读入共享内存
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        a_shared[threadIdx.y][threadIdx.x] = A[i * lda + (bk + threadIdx.x)];
        b_shared[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * ldb + j];
        // 等待块内线程同步
        __syncthreads();

        // 计算块内元素
        for (int k = 0; k < BLOCK_SIZE; k++) {
            c_sub_acc += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        // 再次同步，避免该块内个别线程已经计算完进入下一次循环中，往共享内存写数据，与正在共享内存正在计算中的数据相冲突
        __syncthreads();
    }

    C[i * ldc + j] += c_sub_acc;
}

float MatrixMulCUDA(int version_id, const int M, const int N, const int K,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    float *C, const int ldc) {
    GpuTimer gpu_timer;

    const int block_side_size = 32;
    dim3 threads_per_block(block_side_size, block_side_size);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);
    
    // Warm up.
    GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
      (M, N, K, A, lda, B, ldb, C, ldc);
    cudaMemset(C, 0, sizeof(float) * M * N);

    // Record the start event
    gpu_timer.Start();

    if (version_id == 1) {
      GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
        (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    else if (version_id == 2) {
      GemmKernelv2<block_side_size> << <blocks_per_grid, threads_per_block >> >
        (M, N, K, A, lda, B, ldb, C, ldc);    
    }
    else if (version_id == 3) {
      GemmKernelv3<block_side_size> << <blocks_per_grid, threads_per_block >> >
        (M, N, K, A, lda, B, ldb, C, ldc);    
    }

    // Record the stop event
    gpu_timer.Stop();

    return gpu_timer.ElapsedMillis();
}

#define TEST_CUDA_MODULE_UKERNEL(version_id)                                  \
    do {                                                                      \
        CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice)); \
        CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice)); \
        msec_total = MatrixMulCUDA(version_id, height_a, width_b, width_a, d_a, width_a, d_b, width_b, d_c, width_b); \
        CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost)); \
        printf("gpu version %d -> time: %f s, mean value = %f\n", version_id, msec_total/1000.f, GetMean(h_c, height_a, width_b)); \
    } while (0)

int main() {
    int ret = InitEnvironment(0);
    if (ret != 0) {
        printf("Failed to initialize the environment for cuda.");
        return -1;
    }

    int height_a = 2560, width_a = 800;
    int height_b = 800, width_b = 3200;
    if (width_a != height_b) {
        printf("width_a should be equal to height_b.\n");
        return 1;
    }

    const int mem_size_a = sizeof(float) * height_a * width_a;
    const int mem_size_b = sizeof(float) * height_b * width_b;
    const int mem_size_c = sizeof(float) * height_a * width_b;

    float *h_a = (float *)malloc(mem_size_a);
    float *h_b = (float *)malloc(mem_size_b);
    float *h_c = (float *)malloc(mem_size_c);
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        printf("Fail to malloc.\n");
        return 1;
    }

    // Initialize 
    srand(0);
    GenMatrix(height_a, width_a, h_a);
    GenMatrix(height_b, width_b, h_b);

    // CPU
    time_t t = clock();
    GemmHostV1(height_a, width_b, width_a, h_a, width_a,h_b, width_b, h_c, width_b);
    printf("cpu version 1 -> time: %f s, mean value = %f\n", double(clock() - t)/CLOCKS_PER_SEC, GetMean(h_c, height_a, width_b));
    //MatrixPrint(h_c, height_a, width_b);

    t = clock();
    GemmHostV2(height_a, width_b, width_a, h_a, width_a, h_b, width_b, h_c, width_b);
    printf("cpu version 2 -> time: %f s, mean value = %f\n", double(clock() - t)/CLOCKS_PER_SEC, GetMean(h_c, height_a, width_b));
    //MatrixPrint(h_c, height_a, width_b);

    // GPU
    // Allocate memory in host. 
    float msec_total;
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
    CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
    CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

    TEST_CUDA_MODULE_UKERNEL(1);
    TEST_CUDA_MODULE_UKERNEL(2);
    
    // GPU Device 0: "Tesla T4" with compute capability 7.5 with 40 multi-processors.

    // cpu version 1 -> time: 23.848959 s, mean value = 972596249627025473536.000000
    // cpu version 1 -> time: 29.319994 s, mean value = 972596249627025473536.000000
    // gpu version 1 -> time: 0.040371 s, mean value = 972596249627025473536.000000
    // gpu version 2 -> time: 0.031588 s, mean value = 972596249627025473536.000000
    
    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    CleanUpEnvironment();

    return 0;
}
