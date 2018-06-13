#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

inline static gpuError_t gpuMalloc(void** ptr, size_t size) {
    return (cudaMalloc(ptr, size));
}


#define gpuMalloc cudaMalloc
#define gpuGetDevice cudaGetDevice
#define cuMemAlloc gpuMAlloc
#define gpuMemcpy cuMemcpy
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuFree cudaFree
#define gpuDeviceGetAttribute cuDeviceGetAttribute
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
#define gpuMemset cudaMemset
#define gpuEventRecord cudaEventRecord
#define gpuSetDevice cudaSetDevice
#define gpuEventDestroy cudaEventDestroy
#define gpuMallocHost cudaMallocHost
#define gpuDeviceGetCount cudaDeviceGetCount
#define gpuModuleLaunchKernel cuLaunchKernel
#define cuMemcpyHtoD gpuMemcpyHtoD
#define cuMemcpyDtoH gpuMemcpyDtoH
#define cuCtxDestroy gpuCtxDestroy
#define cuCtxCreate gpuCtxCreate
#define cuCtxSetCurrent gpuCtxSetCurrent
#define cuModuleUnload gpuModuleUnload
#define cuModuleLoadDataEx gpuModuleLoadDataEx
#define cuModuleGetGlobal gpuModuleGetGlobal
#define cuModuleGetFunction gpuModuleGetFunction
#define cuModuleLoadData gpuModuleLoadData

#endif

