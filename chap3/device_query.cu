#include <cuda.h>
#include <stdio.h>
#include <malloc.h>


int main(){
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Num of devices: %d\n", device_count);
    cudaDeviceProp dev_prop;
    for(int i=0; i<device_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        printf("Name: %s\n", dev_prop.name);
        printf("Glob mem: %zu GB\n", dev_prop.totalGlobalMem/1024/1024);
        printf("MaxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
        printf("SMPcount: %d\n", dev_prop.multiProcessorCount);
        printf("MaxBlocksPerSM: %d\n", dev_prop.maxBlocksPerMultiProcessor);
    }

}