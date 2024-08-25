nvcc --optimize 3 -arch=sm_80 -I../3rdparty gemm_fp16_wmma.cu -o a.out
./a.out

# compute-sanitizer ./a.out