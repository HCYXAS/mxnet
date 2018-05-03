/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file gpu_device_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_GPU_DEVICE_STORAGE_H_

#include "mxnet/base.h"
#include "mxnet/storage.h"
#include "../common/cuda_utils.h"
#if MXNET_USE_GPU
#include "gpu_runtime.h"
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
#if MXNET_USE_NCCL
  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#endif  // MXNET_USE_NCCL
  gpuError_t e = gpuMalloc(&ret, size);
  if (e != gpuSuccess && e != gpuErrorCudartUnloading)
    throw std::bad_alloc();
#else   // MXNET_USE_GPU
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_GPU
  return ret;
}

inline void GPUDeviceStorage::Free(void* ptr) {
#if MXNET_USE_GPU
#if MXNET_USE_NCCL
  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#endif  // MXNET_USE_NCCL
  // throw special exception for caller to catch.
  gpuError_t err = gpuFree(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != gpuSuccess && err != gpuErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << gpuGetErrorString(err);
  }
#else   // MXNET_USE_GPU
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_GPU
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
