/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <hip/hip_runtime.h>
//  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <new>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }

  void* Alloc(size_t raw_size) override;
  void Free(void* ptr, size_t raw_size) override;

  void DirectFree(void* ptr, size_t raw_size) override {
    hipError_t err = hipFree(ptr);
    size_t size = raw_size + NDEV;
    // ignore unloading error, as memory has already been recycled
    if (err != hipSuccess) {
        /*TODO.Need to revisit: unknown error is reported in HIP/CUDA path*/
      //LOG(FATAL) << "CUDA: " << hipGetErrorString(err);
    }
    used_memory_ -= size;
  }

 private:
  void ReleaseAll();
  // internal mutex
  std::mutex mutex_;
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // number of devices
  const int NDEV = 32;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void* GPUPooledStorageManager::Alloc(size_t raw_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t size = raw_size + NDEV;
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    size_t free, total;
    hipMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100)
      ReleaseAll();

    void* ret = nullptr;
    hipError_t e = hipMalloc(&ret, size);
    if (e != hipSuccess) {
      LOG(FATAL) << "cudaMalloc failed: " << hipGetErrorString(e);
    }
    used_memory_ += size;
    return ret;
  } else {
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    return ret;
  }
}

void GPUPooledStorageManager::Free(void* ptr, size_t raw_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t size = raw_size + NDEV;
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(ptr);
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      DirectFree(j, i.first - NDEV);
    }
  }
  memory_pool_.clear();
}
#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
