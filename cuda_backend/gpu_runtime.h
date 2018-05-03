#ifndef CUDA_BACKEND_GPU_RUNTIME_H
#define CUDA_BACKEND_GPU_RUNTIME_H


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define gpuMalloc 							cudaMalloc
#define gpuGetDevice 							cudaGetDevice
#define gpuMemAlloc 							cuMemAlloc
#define gpuMemcpy							cudaMemcpy
#define gpuMemcpyHtoD 							cuMemcpyHtoD
#define gpuMemcpyDtoH 							cuMemcpyDtoH
#define gpuMemcpyAsync 							cudaMemcpyAsync
#define gpuMemcpypeerAsync 						cudaMemcpypeerAsync
#define gpuFree 							cudaFree
#define gpuMemcpyPeerAsync						cudaMemcpyPeerAsync
#define gpuMemcpy2DAsync						cudaMemcpy2DAsync
#define gpuDevAttrMultiProcessorCount 					cudaDevAttrMultiProcessorCount
#define gpuMemGetInfo 							cudaMemGetInfo
#define gpuFreeHost 							cudaFreeHost
#define gpuMemset 							cudaMemset
#define gpuMemsetAsync							cudaMemsetAsync
#define gpuMemFree							cuMemFree
#define gpuHostAllocPortable						cudaHostAllocPortable
#define gpuMemsetD32 							cudaMemsetD32
#define gpuMemcpyDeviceToHost                				cudaMemcpyDeviceToHost
#define gpuMemcpyKind                     				cudaMemcpyKind
#define gpuMemcpyDeviceToDevice          				cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice           				cudaMemcpyHostToDevice
#define gpuDeviceSynchronize 						cudaDeviceSynchronize
#define gpuDeviceSetSharedMemConfig 					cudaDeviceSetSharedMemConfig
#define gpuGetErrorString 						cudaGetErrorString
#define gpuSetDevice 							cudaSetDevice
#define gpuEventRecord 							cudaEventRecord
#define gpuEventDestroy 						cudaEventDestroy
#define gpuEventQuery 							cudaEventQuery
#define gpuEventCreateWithFlags 					cudaEventCreateWithFlags
#define gpuMallocHost 							cudaMallocHost
#define gpuHostAlloc							cudaHostAlloc
#define gpuMallocPitch							cudaMallocPitch
#define gpuDeviceGetCount 						cuDeviceGetCount
#define gpuGetDeviceCount 						cudaGetDeviceCount
#define gpuDeviceCanAccessPeer						cudaDeviceCanAccessPeer
#define gpuDeviceEnableAccessPeer					cudaDeviceEnableAccessPeer
#define gpuGetDeviceProperties 						cudaGetDeviceProperties
#define gpudeviptr 							CUdeviceptr
#define gpuDeviceEnablePeerAccess   					cudaDeviceEnablePeerAccess
#define gpuCtxDestroy 							cuCtxDestroy
#define gpuCtxCreate 							cuCtxCreate 
#define gpuCtxSetCurrent 						cuCtxSetCurrent
#define gpuModuleUnload 						cuModuleUnload 
#define gpuModuleLoadDataEx 						cuModuleLoadDataEx
#define gpuModuleGetGlobal 						cuModuleGetGlobal
#define gpuModuleGetFunction 						cuModuleGetFunction
#define gpuModuleLoadData 						cuModuleLoadData
#define gpuDeviceAttributeMultiProcessorCount  				cudaDevAttrMultiProcessorCount
#define gpuDevAttrComputeCapabilityMajor 	   			cudaDevAttrComputeCapabilityMajor
#define gpuDevAttrComputeCapabilityMinor 	   			cudaDevAttrComputeCapabilityMinor
#define gpuDevAttrGlobalMemoryBusWidth	   	   			cudaDevAttrGlobalMemoryBusWidth
#define gpuDevAttrMemoryClockRate					cudaDevAttrMemoryClockRate
#define gpuDevAttrMaxThreadsPerBlock		   			cudaDevAttrMaxThreadsPerBlock
#define gpuDevAttrWarpSize						cudaDevAttrWarpSize
#define gpuThreadSynchronize						cudaThreadSynchronize
#define gpuFuncGetAttributes						cudaFuncGetAttributes	
#define gpuLaunchKernel 						cuLaunchKernel
#define gpuGetLastError 						cudaGetLastError 
#define gpuresult 							CUresult
#define gpuErrorDeinitialized  						CUDA_ERROR_DEINITIALIZED
#define gpu_SUCCESS 							CUDA_SUCCESS
#define gpuError 							cudaError
#define gpuError_t                  					cudaError_t
#define gpuErrorNotSupported        					cudaErrorNotSupported
#define gpuErrorNotReady						cudaErrorNotReady
#define gpuErrorMemoryAllocation					cudaErrorMemoryAllocation  
#define gpuErrorInvalidValue						cudaErrorInvalidValue	
#define gpuErrorCudartUnloading     					cudaErrorCudartUnloading
#define gpuErrorInvalidConfiguration					cudaErrorInvalidConfiguration 
#define gpuErrorPeerAccessAlreadyEnabled 				cudaErrorPeerAccessAlreadyEnabled 
#define gpuStream_t 							cudaStream_t
#define gpuSuccess 							cudaSuccess
#define gpuDataType_t							cudaDataType_t
#define gpuDeviceProp_t             					cudaDeviceProp_t
#define gpuGetErrorName							cudaGetErrorName
#define gpuStream							CUStream
#define	gpufunction 							CUfunction
#define gpuStreamQuery 							cudaStreamQuery
#define gpuStreamDestroy 						cudaStreamDestroy
#define gpuStreamCreate 						cudaStreamCreate
#define gpuStreamSynchronize						cudaStreamSynchronize
#define gpuPeekAtLastError 						cudaPeekAtLastError
#define gpuTimerMalloc 							cudaTimerMalloc
#define gpuInit 							cudaInit
#define gpuDeviceGetAttribute    					cudaDeviceGetAttribute
#define gpuUnbindTexture						cudaUnbindTexture
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor  			cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuBindTexture							cudaBindTexture
#define	gpuSharedMemBankSizeEightByte					cudaSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte					cudaSharedMemBankSizeFourByte
#define gpuDeviceProp 					 		cudaDeviceProp



#endif
