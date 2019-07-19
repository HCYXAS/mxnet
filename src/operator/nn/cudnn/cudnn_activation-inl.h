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
 * \file cudnn_activation-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_ACTIVATION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_ACTIVATION_INL_H_
#include <algorithm>
#include <vector>
#include "../activation-inl.h"

namespace mxnet {
namespace op {
template<typename DType>
class CuDNNActivationOp {
 public:
  CuDNNActivationOp() {
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    CUDNN_CALL(miopenCreateActivationDescriptor(&desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&shape_desc_));
  }

  void Init(const ActivationParam &param) {
    param_ = param;
    switch (param_.act_type) {
      case activation::kReLU:
        mode_ = miopenActivationRELU;
        break;
      case activation::kSigmoid:
        mode_ = miopenActivationLOGISTIC;
        break;
      case activation::kTanh:
        mode_ = miopenActivationTANH;
        break;
      default:
        LOG(FATAL) << "Not implmented";
        break;
    }
    double alpha = 1.0f;
    double beta  = 1.0f;
    relu_ceil_   = 1.0f;
    CUDNN_CALL(miopenSetActivationDescriptor(desc_, mode_, alpha, beta, relu_ceil_));//TODO Temporary fix for input parameters
  }

  ~CuDNNActivationOp() {
    CUDNN_CALL(miopenDestroyTensorDescriptor(shape_desc_));
    CUDNN_CALL(miopenDestroyActivationDescriptor(desc_));
  }

  void Forward(const OpContext &ctx, const TBlob &in_data,
      const OpReqType &req, const TBlob &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> data;
    Tensor<gpu, 4, DType> out;
    if (in_data.ndim() == 2) {
      Shape<4> dshape = Shape4(in_data.shape_[0],
                               in_data.shape_[1], 1, 1);
      data = in_data.get_with_shape<gpu, 4, DType>(dshape, s);
      out = out_data.get_with_shape<gpu, 4, DType>(dshape, s);
    } else {
      Shape<4> dshape;
      index_t size_left = in_data.Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_data.ndim()) {
          dshape[i] = in_data.shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      data = in_data.get_with_shape<gpu, 4, DType>(dshape, s);
      out = out_data.get_with_shape<gpu, 4, DType>(dshape, s);
    }
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(miopenSet4dTensorDescriptor(shape_desc_,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]));
    CUDNN_CALL(miopenActivationForward(s->dnn_handle_,
                                     desc_,
                                     &alpha,
                                     shape_desc_,
                                     data.dptr_,
                                     &beta,
                                     shape_desc_,
                                     out.dptr_));
  }

  // backward computation for cudnn activation operator. Note that for relu
  // it's okay to pass "out_data" as "in_data", since it doesn't make any
  // difference in terms of computing the gradient of relu.
  void Backward(const OpContext &ctx, const TBlob &out_grad,
      const TBlob &in_data, const TBlob &out_data,
      const OpReqType &req, const TBlob &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> grad;
    Tensor<gpu, 4, DType> data;
    Tensor<gpu, 4, DType> output_data;
    Tensor<gpu, 4, DType> input_grad;
    if (in_grad.ndim() == 2) {
      Shape<4> dshape = Shape4(in_grad.shape_[0],
                               in_grad.shape_[1], 1, 1);
      data = in_data.get_with_shape<gpu, 4, DType>(dshape, s);
      grad = out_grad.get_with_shape<gpu, 4, DType>(dshape, s);
      output_data = out_data.get_with_shape<gpu, 4, DType>(dshape, s);
      input_grad = in_grad.get_with_shape<gpu, 4, DType>(dshape, s);
    } else {
      Shape<4> dshape;
      index_t size_left = in_grad.Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_grad.ndim()) {
          dshape[i] = in_grad.shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      data = in_data.get_with_shape<gpu, 4, DType>(dshape, s);
      output_data = out_data.get_with_shape<gpu, 4, DType>(dshape, s);
      grad = out_grad.get_with_shape<gpu, 4, DType>(dshape, s);
      input_grad = in_grad.get_with_shape<gpu, 4, DType>(dshape, s);
    }
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(miopenSet4dTensorDescriptor(shape_desc_,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]));
    CUDNN_CALL(miopenActivationBackward(s->dnn_handle_,
                                       desc_,
                                       &alpha,
                                       shape_desc_,
                                       output_data.dptr_,
                                       shape_desc_,
                                       grad.dptr_,
                                       shape_desc_,
                                       data.dptr_,
                                       &beta,
                                       shape_desc_,
                                       input_grad.dptr_));
  }

 private:
  miopenDataType_t dtype_;
  miopenActivationMode_t mode_;
  miopenTensorDescriptor_t shape_desc_;
  ActivationParam param_;
  miopenActivationDescriptor_t desc_;
  double relu_ceil_;
};  // class CuDNNActivationOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_ACTIVATION_INL_H_
