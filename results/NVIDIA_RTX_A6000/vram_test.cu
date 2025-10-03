#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    const size_t MB = 256;
    const int iters = 50;
    for (int i=0; i<iters; i++) {
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, MB*1024*1024);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed at iter %d: %s\n", i, cudaGetErrorString(err));
            return 1;
        }
        cudaMemset(ptr, 0xA5, MB*1024*1024);
        cudaFree(ptr);
    }
    printf("VRAM allocation/free test completed successfully.\n");
    return 0;
}
