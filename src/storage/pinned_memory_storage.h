/*!
 * Copyright (c) 2015 by Contributors
 * \file cpu_device_storage.h
 * \brief CPU storage with pinned memory
 */
#ifndef MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
#define MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
#if MXNET_USE_GPU

#include <dmlc/logging.h>
#include "mxnet/base.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace storage {

class PinnedMemoryStorage {
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
};

inline void* PinnedMemoryStorage::Alloc(size_t size) {
  void* ret = nullptr;
  // make the memory available across all devices
  CUDA_CALL(hipHostMalloc(&ret, size, hipHostMallocPortable));
  return ret;
}

inline void PinnedMemoryStorage::Free(void* ptr) {
  hipError_t err = hipHostFree(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != hipSuccess) {
    LOG(FATAL) << "CUDA: " << hipGetErrorString(err);
  }
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_USE_GPU
#endif  // MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
