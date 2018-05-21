/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn.cu
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_CUDNN == 1 || MXNET_USE_MIOPEN == 1
#include "./cudnn_lrn-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(LRNParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 || MXNET_USE_MIOPEN == 1
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNLocalResponseNormOp<DType>(param);
  })
#else
#ifdef __HIP_PLATFORM_NVCC__ 
#if CUDA_VERSION == 7000 && MXNET_USE_CUDNN == 1
  LOG(FATAL) << "Due to old CUDA compiler bug, LRN is disabled."
             << "Please upgrade CUDA to 7.5+ or use CUDNN";
#endif
#else
  op = new LocalResponseNormOp<gpu>(param);
#endif  // CUDA_VERSION
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet


