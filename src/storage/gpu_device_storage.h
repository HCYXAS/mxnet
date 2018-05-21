/*!
 * Copyright (c) 2015 by Contributors
 * \file gpu_device_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_GPU_DEVICE_STORAGE_H_

#include "mxnet/base.h"
#include "../common/cuda_utils.h"
#if MXNET_USE_GPU
#include <hip/hip_runtime.h>
#endif  // MXNET_USE_GPU
#include <new>

namespace mxnet {
namespace storage {

/*!
 * \brief GPU storage implementation.
 */
class GPUDeviceStorage {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);
};  // class GPUDeviceStorage

inline void* GPUDeviceStorage::Alloc(size_t size) {
  void* ret = nullptr;
#if MXNET_USE_GPU
  hipError_t e = hipMalloc(&ret, size);
  if (e != hipSuccess && e != hipErrorCudartUnloading)
    throw std::bad_alloc();
#else   // MXNET_USE_GPU
  LOG(FATAL) << "Please compile with GPU enabled";
#endif  // MXNET_USE_GPU
  return ret;
}

inline void GPUDeviceStorage::Free(void* ptr) {
#if MXNET_USE_GPU
  // throw special exception for caller to catch.
  hipError_t err = hipFree(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != hipSuccess && err != hipErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << hipGetErrorString(err);
  }
#else   // MXNET_USE_GPU
  LOG(FATAL) << "Please compile with GPU enabled";
#endif  // MXNET_USE_GPU
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
