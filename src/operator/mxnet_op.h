/*!
 * Copyright (c) 2017 by Contributors
 * \file mxnet_op.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_MXNET_OP_H_
#define MXNET_OPERATOR_MXNET_OP_H_

#include <mxnet/base.h>
#include <algorithm>

#include <hip/hip_runtime.h>
namespace mxnet {
namespace op {
namespace mxnet_op {
using namespace mshadow;

#if __HIP_DEVICE_COMPILE__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif


#ifdef __HIPCC__
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; \
      i < (n); \
      i += hipBlockDim_x * hipGridDim_x)


/*!
 * \brief Get the number of blocks for cuda kernel given N
 */
inline int cuda_get_num_blocks(const int N) {
  using namespace mshadow::cuda;
  return std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
}
#endif  // __HIPCC__


/*! \brief operator request type switch */
#define MXNET_ASSIGN_REQ_SWITCH(req, ReqType, ...)  \
  switch (req) {                                    \
  case kNullOp:                                     \
    break;                                          \
  case kWriteInplace:                               \
  case kWriteTo:                                    \
    {                                               \
      const int ReqType = kWriteTo;                 \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kAddTo:                                      \
    {                                               \
      const int ReqType = kAddTo;                   \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    break;                                          \
  }


/*!
 * \brief assign the val to out according
 * to request in Kernel::Launch
 * \param out the data to be assigned
 * \param req the assignment request
 * \param val the value to be assigned to out
 * \tparam OType output type
 * \tparam VType value type
 */
#define KERNEL_ASSIGN(out, req, val)  \
  {                                   \
    switch (req) {                    \
      case kNullOp:                   \
        break;                        \
      case kWriteTo:                  \
      case kWriteInplace:             \
        (out) = (val);                \
        break;                        \
      case kAddTo:                    \
        (out) += (val);               \
        break;                        \
      default:                        \
        break;                        \
    }                                 \
  }


/* \brief Compute flattened index given coordinates and shape. */
template<int ndim>
MSHADOW_XINLINE int ravel(const Shape<ndim>& coord, const Shape<ndim>& shape) {
  int ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > coord[i]) * coord[i];
  }
  return ret;
}


/* Compute coordinates from flattened index given shape */
template<int ndim>
MSHADOW_XINLINE Shape<ndim> unravel(const int idx, const Shape<ndim>& shape) {
  Shape<ndim> ret;
  #pragma unroll
  for (int i = ndim-1, j = idx; i >=0; --i) {
    int tmp = j / shape[i];
    ret[i] = j - tmp*shape[i];
    j = tmp;
  }
  return ret;
}


/* Compute dot product of two vector */
template<int ndim>
MSHADOW_XINLINE int dot(const Shape<ndim>& coord, const Shape<ndim>& stride) {
  int ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i)
    ret += coord[i] * stride[i];
  return ret;
}


/* Combining unravel and dot */
template<int ndim>
MSHADOW_XINLINE int unravel_dot(const int idx, const Shape<ndim>& shape,
  const Shape<ndim>& stride) {
  int ret = 0;
  #pragma unroll
  for (int i = ndim-1, j = idx; i >=0; --i) {
    int tmp = j / shape[i];
    ret += (j - tmp*shape[i])*stride[i];
    j = tmp;
  }
  return ret;
}


/* Calculate stride of each dim from shape */
template<int ndim>
MSHADOW_XINLINE Shape<ndim> calc_stride(const Shape<ndim>& shape) {
  Shape<ndim> stride;
  index_t cumprod = 1;
  #pragma unroll
  for (int i = ndim - 1; i >= 0; --i) {
    stride[i] = (shape[i] > 1) ? cumprod : 0;
    cumprod *= shape[i];
  }
  return stride;
}


struct fill {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType val) {
    out[i] = val;
  }
};


struct set_zero {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out) {
    out[i] = static_cast<DType>(0);
  }
};


template<typename OP, typename xpu>
struct Kernel;


template<typename OP>
struct Kernel<OP, cpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<cpu> *s, int N, Args... args) {
#if (MXNET_USE_GPU == 0)
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
  }
};


#ifdef __HIPCC__
template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel(int N, Args... args) {
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x) {
    OP::Map(i, args...);
  }
}


template<typename OP>
struct Kernel<OP, gpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mxnet_generic_kernel<OP, Args...>), dim3(ngrid), dim3(kBaseThreadNum), 0, mshadow::Stream<gpu>::GetStream(s),N, args...);
  }
};
#endif  // __HIPCC__


}  // namespace mxnet_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MXNET_OP_H_
