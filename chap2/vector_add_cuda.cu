#include<cuda.h>
#include<stdio.h>
#include<malloc.h>


__global__ void vecAddKernel(float* A, float* B, float* C, int len){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len){
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int len){
    int size = sizeof(float) * len;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size);

    vecAddKernel<<<ceil(len/256.0), 256>>>(d_A, d_B, d_C, len);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


int main(){
    int n = 270;
    float *a = (float*) malloc(sizeof(float) * n);
    float *b = (float*) malloc(sizeof(float) * n);
    float *c = (float*) malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = i*2;
        c[i] = 0;
    }
    vecAdd(a, b, c, n);
    for (int i = 0; i < n; i++)
    {
        printf("%.2f, ", c[i]);
    }
    printf("\n");
}
