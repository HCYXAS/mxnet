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
 * Copyright (c) 2017 by Contributors
 * \file quantized_pooling.cu
*/
#include <mxnet/operator_util.h>
#include <vector>
#include "../nn/pooling-inl.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

#if MXNET_USE_MIOPEN == 1  && CUDA_VERSION >= 8000
template<typename DType>
class QuantizedCuDNNPoolingOp {
 public:
  QuantizedCuDNNPoolingOp() {
    MIOPEN_CALL(miopenCreatePoolingDescriptor(&pool_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&in_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&out_desc_));
  }

  void Init(const PoolingParam& param, const TShape& dshape, const TShape& oshape) {
    const int N = 0, H = 2, W = 3, C = 1;
    const miopenDataType_t dtype = mshadow::DataType<DType>::kCudnnFlag;
    CHECK(param.kernel.ndim() == 2) << "Only support 2D pooling";
    if (param.pool_type == pool_enum::kMaxPooling) {
      mode_ = miopenPoolingMax;
    } else if (param.pool_type == pool_enum::kAvgPooling) {
      mode_ = miopenPoolingAverage; 
    } else {
      LOG(FATAL) << "QuantizedCuDNNPoolingOp only supports pool_type=max/avg";
    }
    MIOPEN_CALL(miopenSet4dTensorDescriptor(in_desc_,
                                          dtype,
                                          dshape[N],
                                          dshape[C],
                                          dshape[H],
                                          dshape[W]));
    MIOPEN_CALL(miopenSet4dTensorDescriptor(out_desc_,
                                          dtype,
                                          oshape[N],
                                          oshape[C],
                                          oshape[H],
                                          oshape[W]));

   MIOPEN_CALL(miopenSet2dPoolingDescriptor(pool_desc_,
                                               mode_,
                                               param.global_pool ? dshape[2] : param.kernel[0],
                                               param.global_pool ? dshape[3] : param.kernel[1],
                                               param.pad[0],
                                               param.pad[1],
                                               param.global_pool ? 1 : param.stride[0],
                                               param.global_pool ? 1 : param.stride[1]));

  workspaceSize = 0;
   MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(out_desc_, &workspaceSize));
  if(workspaceSize > 0)
   hipMalloc(&workspace, workspaceSize);

  }

  ~QuantizedCuDNNPoolingOp() {
    MIOPEN_CALL(miopenDestroyTensorDescriptor(in_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(out_desc_));
    MIOPEN_CALL(miopenDestroyPoolingDescriptor(pool_desc_));
    hipFree(workspace);
  }

  void Forward(mshadow::Stream<gpu>* s,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs) {
    CHECK_EQ(inputs.size(), 3U);
    CHECK_EQ(outputs.size(), 3U);
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t temp_workspaceSize = 0;
      MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(out_desc_, &temp_workspaceSize));
      if (temp_workspaceSize > 0 && temp_workspaceSize > workspaceSize ) {
            workspaceSize = temp_workspaceSize;
            hipFree(workspace);
            hipMalloc(&workspace, workspaceSize);
    }

    MIOPEN_CALL(miopenPoolingForward(s->dnn_handle_,
                                     pool_desc_,
                                     &alpha,
                                     in_desc_,
                                     inputs[0].dptr_,
                                     &beta,
                                     out_desc_,
                                     outputs[0].dptr_,
                                     true,
                                     workspace,
                                     workspaceSize));

    Tensor<gpu, 1, float> omin_range = outputs[1].FlatTo1D<gpu, float>(s);
    Tensor<gpu, 1, float> omax_range = outputs[2].FlatTo1D<gpu, float>(s);
    ASSIGN_DISPATCH(omin_range, req[1],
      F<mshadow_op::identity>(inputs[1].FlatTo1D<gpu, float>(s)));
    ASSIGN_DISPATCH(omax_range, req[2],
      F<mshadow_op::identity>(inputs[2].FlatTo1D<gpu, float>(s)));
  }

 private:
  miopenPoolingMode_t mode_;
  miopenTensorDescriptor_t in_desc_;
  miopenTensorDescriptor_t out_desc_;
  miopenPoolingDescriptor_t pool_desc_;

  size_t workspaceSize;
  void* workspace;
};  // class QuantizedCuDNNPoolingOp
#endif  // MXNET_USE_MIOPEN == 1  CUDA_VERSION >= 8000

void QuantizedPoolingForwardGPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(param.kernel.ndim(), 2U)
    << "QuantizedPoolingForward<gpu> only supports 2D convolution for now";
#if MXNET_USE_MIOPEN == 1  && CUDA_VERSION >= 8000
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local QuantizedCuDNNPoolingOp<int8_t> op;
#else
  static MX_THREAD_LOCAL QuantizedCuDNNPoolingOp<int8_t> op;
#endif  // DMLC_CXX11_THREAD_LOCAL
  op.Init(param, {inputs[0].shape_}, {outputs[0].shape_});
  op.Forward(ctx.get_stream<gpu>(), inputs, req, outputs);
#else
  LOG(FATAL) << "QuantizedPoolingForward<gpu> only supports cudnnPoolingForward "
                "with CUDNN >= 6.0 and CUDA >= 8.0";
#endif  // MXNET_USE_MIOPEN == 1  && CUDA_VERSION >= 8000
}

NNVM_REGISTER_OP(_contrib_quantized_pooling)
.set_attr<FCompute>("FCompute<gpu>", QuantizedPoolingForwardGPU);

}  // namespace op
}  // namespace mxnet
