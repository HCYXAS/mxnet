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

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_SOFTMAX_ACTIVATION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_SOFTMAX_ACTIVATION_INL_H_
#include <algorithm>
#include <vector>
#include "../softmax_activation-inl.h"

namespace mxnet {
namespace op {
class CuDNNSoftmaxActivationOp {
 public:
  CuDNNSoftmaxActivationOp() {
    dtype_ = miopenFloat;
    MIOPEN_CALL(miopenCreateTensorDescriptor(&shape_desc_));
  }

  void Init(SoftmaxActivationParam param) {
    this->param_ = param;
  }

  ~CuDNNSoftmaxActivationOp() {
    MIOPEN_CALL(miopenDestroyTensorDescriptor(shape_desc_));
  }

  void Forward(const OpContext &ctx, const TBlob &in_data,
               const OpReqType &req, const TBlob &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> data;
    Tensor<gpu, 4> out;
    miopenSoftmaxMode_t softmax_mode;
    if (param_.mode == softmax_activation::kInstance) {
      CHECK_EQ(in_data.ndim(), 2)
        << "Input need to have 2 dimensions when mode=instance.";
      Shape<4> dshape = Shape4(in_data.shape_[0], in_data.shape_[1], 1, 1);
      data = in_data.get_with_shape<gpu, 4, real_t>(dshape, s);
      out = out_data.get_with_shape<gpu, 4, real_t>(dshape, s);
      softmax_mode =  MIOPEN_SOFTMAX_MODE_INSTANCE;
    } else {
      CHECK_GE(in_data.ndim(), 3)
        << "Input need to have a least 3 dimensions when mode=channel";
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
      data = in_data.get_with_shape<gpu, 4, real_t>(dshape, s);
      out = out_data.get_with_shape<gpu, 4, real_t>(dshape, s);
      softmax_mode = MIOPEN_SOFTMAX_MODE_CHANNEL;
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    MIOPEN_CALL(miopenSet4dTensorDescriptor(shape_desc_,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]));
    MIOPEN_CALL(miopenSoftmaxForward_V2(s->dnn_handle_,
                                   &alpha,
                                   shape_desc_,
                                   data.dptr_,
                                   &beta,
                                   shape_desc_,
                                   out.dptr_,
                                  MIOPEN_SOFTMAX_ACCURATE,
                                  softmax_mode));

  }

  void Backward(const OpContext &ctx, const TBlob &out_grad,
                const TBlob &out_data, const OpReqType &req,
                const TBlob &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    float alpha = 1.0f;
    float beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> grad;
    Tensor<gpu, 4> output_data;
    Tensor<gpu, 4> input_grad;
    //TODO MIOpen does not support Softmax modes. MIOpen implements the SOFTMAX_MODE_CHANNEL flavor.
    //cudnnSoftmaxMode_t softmax_mode;
    miopenSoftmaxMode_t softmax_mode;
    if (param_.mode == softmax_activation::kInstance) {
      CHECK_EQ(in_grad.ndim(), 2)
        << "Input need to have 2 dimensions when mode=instance.";
      Shape<4> dshape = Shape4(in_grad.shape_[0], in_grad.shape_[1], 1, 1);
      grad = out_grad.get_with_shape<gpu, 4, real_t>(dshape, s);
      output_data = out_data.get_with_shape<gpu, 4, real_t>(dshape, s);
      input_grad = in_grad.get_with_shape<gpu, 4, real_t>(dshape, s);
      softmax_mode =  MIOPEN_SOFTMAX_MODE_INSTANCE;
    } else {
      CHECK_GE(in_grad.ndim(), 3)
        << "Input need to have a least 3 dimensions when mode=channel";
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
      output_data = out_data.get_with_shape<gpu, 4, real_t>(dshape, s);
      grad = out_grad.get_with_shape<gpu, 4, real_t>(dshape, s);
      input_grad = in_grad.get_with_shape<gpu, 4, real_t>(dshape, s);
      softmax_mode = MIOPEN_SOFTMAX_MODE_CHANNEL;
    }
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    MIOPEN_CALL(miopenSet4dTensorDescriptor(shape_desc_,
                                          dtype_,
                                          input_grad.shape_[0],
                                          input_grad.shape_[1],
                                          input_grad.shape_[2],
                                          input_grad.shape_[3]));
    MIOPEN_CALL(miopenSoftmaxBackward_V2(s->dnn_handle_,
                                    &alpha,
                                    shape_desc_,
                                    output_data.dptr_,
                                    shape_desc_,
                                    grad.dptr_,
                                    &beta,
                                    shape_desc_,
                                    input_grad.dptr_,
                                   MIOPEN_SOFTMAX_ACCURATE,
                                    softmax_mode));


  }

 private:
  miopenDataType_t dtype_;
  miopenTensorDescriptor_t shape_desc_;
  SoftmaxActivationParam param_;
};  // class CuDNNSoftmaxActivationOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_SOFTMAX_ACTIVATION_INL_H_
