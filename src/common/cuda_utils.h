/*!
 * Copyright (c) 2015 by Contributors
 * \file cuda_utils.h
 * \brief CUDA debugging utilities.
 */
#ifndef MXNET_COMMON_CUDA_UTILS_H_
#define MXNET_COMMON_CUDA_UTILS_H_


#include <dmlc/logging.h>
#include <mshadow/base.h>

#if MXNET_USE_GPU
#include <hip-wrappers.h> // dummy include file placed in /opt/rocm/include
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hiprand.h>

namespace mxnet {
namespace common {
/*! \brief common utils for cuda */
namespace cuda {
/*!
 * \brief Get string representation of hipBLAS errors.
 * \param error The error.
 * \return String representation.
 */
inline const char* HipblasGetErrorString(hipblasStatus_t error) {
  switch (error) {
  case HIPBLAS_STATUS_SUCCESS:
    return "HIPBLAS_STATUS_SUCCESS";
  case HIPBLAS_STATUS_NOT_INITIALIZED:
    return "HIPBLAS_STATUS_NOT_INITIALIZED";
  case HIPBLAS_STATUS_ALLOC_FAILED:
    return "HIPBLAS_STATUS_ALLOC_FAILED";
  case HIPBLAS_STATUS_INVALID_VALUE:
    return "HIPBLAS_STATUS_INVALID_VALUE";
//  case HIPBLAS_STATUS_ARCH_MISMATCH: // NOT SUPPORTED YET
//    return "HIPBLAS_STATUS_ARCH_MISMATCH";
  case HIPBLAS_STATUS_MAPPING_ERROR:
    return "HIPBLAS_STATUS_MAPPING_ERROR";
  case HIPBLAS_STATUS_EXECUTION_FAILED:
    return "HIPBLAS_STATUS_EXECUTION_FAILED";
  case HIPBLAS_STATUS_INTERNAL_ERROR:
    return "HIPBLAS_STATUS_INTERNAL_ERROR";
  case HIPBLAS_STATUS_NOT_SUPPORTED:
    return "HIPBLAS_STATUS_NOT_SUPPORTED";
  default:
    break;
  }
  return "Unknown hipBLAS status";
}

/*!
 * \brief Get string representation of hipRNG errors.
 * \param status The status.
 * \return String representation.
 */
inline const char* HiprandGetErrorString(hiprandStatus_t status) {
  switch (status) {
  case HIPRAND_STATUS_SUCCESS:
    return "HIPRAND_STATUS_SUCCESS";
  case HIPRAND_STATUS_VERSION_MISMATCH:
    return "HIPRAND_STATUS_VERSION_MISMATCH";
  case HIPRAND_STATUS_NOT_INITIALIZED:
    return "HIPRAND_STATUS_NOT_INITIALIZED";
  case HIPRAND_STATUS_ALLOCATION_FAILED:
    return "HIPRAND_STATUS_ALLOCATION_FAILED";
  case HIPRAND_STATUS_TYPE_ERROR:
    return "HIPRAND_STATUS_TYPE_ERROR";
  case HIPRAND_STATUS_OUT_OF_RANGE:
    return "HIPRAND_STATUS_OUT_OF_RANGE";
  case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
//  case HIPRNG_STATUS_DOUBLE_PRECISION_REQUIRED: // NOT SUPPORTED YET
//    return "HIPRNG_STATUS_DOUBLE_PRECISION_REQUIRED";
  case HIPRAND_STATUS_LAUNCH_FAILURE:
    return "HIPRAND_STATUS_LAUNCH_FAILURE";
  case HIPRAND_STATUS_PREEXISTING_FAILURE:
    return "HIPRAND_STATUS_PREEXISTING_FAILURE";
  case HIPRAND_STATUS_INITIALIZATION_FAILED:
    return "HIPRAND_STATUS_INITIALIZATION_FAILED";
  case HIPRAND_STATUS_ARCH_MISMATCH:
    return "HIPRAND_STATUS_ARCH_MISMATCH";
  case HIPRAND_STATUS_INTERNAL_ERROR:
    return "HIPRAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown hipRAND status";
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    hipError_t e = hipGetLastError();                                      \
    CHECK_EQ(e, hipSuccess) << (msg) << " CUDA: " << hipGetErrorString(e); \
  }

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                            \
  {                                                                \
    hipError_t e = (func);                                        \
    CHECK(e == hipSuccess)       \
        << "CUDA: " << hipGetErrorString(e);                      \
  }

/*!
 * \brief Protected hipBLAS call.
 * \param func Expression to call.
 *
 * It checks for hipBLAS errors after invocation of the expression.
 */
#define HIPBLAS_CALL(func)                                       \
  {                                                             \
    hipblasStatus_t e = (func);                                  \
    CHECK_EQ(e, HIPBLAS_STATUS_SUCCESS)                          \
        << "hipBLAS: " << common::cuda::HipblasGetErrorString(e); \
  }

/*!
 * \brief Protected hipRNG call.
 * \param func Expression to call.
 *
 * It checks for hipRNG errors after invocation of the expression.
 */
#define HIPRNG_CALL(func)                                       \
  {                                                             \
    hiprandStatus_t e = (func);                                  \
    CHECK_EQ(e, HIPRNG_STATUS_SUCCESS)                          \
        << "hipRNG: " << common::cuda::HiprandGetErrorString(e); \
  }

#endif  // MXNET_USE_GPU

#if MXNET_USE_MIOPEN 

#include <miopen/miopen.h>

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    miopenStatus_t  e = (func);                                                 \
    CHECK_EQ(e, miopenStatusSuccess) << "miopen error code: " << e; \
  }

#endif

#if MXNET_USE_CUDNN 
#include <cudnn.h>

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    cudnnStatus_t  e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN:" << cudnnGetErrorString(e); \
  }

#endif  // MXNET_USE_CUDNN

// Overload atomicAdd to work for floats on all architectures
//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
//#if (__HIP_DEVICE_COMPILE__) && (__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__)
#if (__HIP_DEVICE_COMPILE__ && (__CUDA_ARCH__ < 600)) || defined(__HCC__)
static inline __device__  void atomicAdd(double *address, double val) {
  unsigned long long* address_as_ull =                  // NOLINT(*)
    reinterpret_cast<unsigned long long*>(address);     // NOLINT(*)
  unsigned long long old = *address_as_ull;             // NOLINT(*)
  unsigned long long assumed;                           // NOLINT(*)

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}
#endif

// Overload atomicAdd for half precision
// Taken from:
// https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
//#if defined(__CUDA_ARCH__)
//#if (__HIP_DEVICE_COMPILE__)
#if (__HIP_DEVICE_COMPILE__) || defined(__HCC__)
static inline __device__ void atomicAdd(mshadow::half::half_t *address,
                                        mshadow::half::half_t val) {
  unsigned int *address_as_ui =
      reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) -
                                   (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    mshadow::half::half_t hsum;
    hsum.half_ =
        reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;
    old = reinterpret_cast<size_t>(address) & 2
              ? (old & 0xffff) | (hsum.half_ << 16)
              : (old & 0xffff0000) | hsum.half_;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

#endif  // MXNET_COMMON_CUDA_UTILS_H_
