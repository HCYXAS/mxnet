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
 * \file cudnn_pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_
#include <algorithm>
#include <vector>
#include "../pooling-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class CuDNNPoolingOp {
 public:
  CuDNNPoolingOp() {
    // TODO(xxx): fp16
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    MIOPEN_CALL(miopenCreatePoolingDescriptor(&pooling_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&in_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&out_desc_));
    workspaceSize =0 ;
    workspace = nullptr;

  }

  void Init(const PoolingParam &p) {
    param_ = p;
    switch (param_.pool_type) {
      case pool_enum::kMaxPooling:
        mode_ = miopenPoolingMax;
        break;
      case pool_enum::kAvgPooling:
        if (param_.count_include_pad.has_value() && !param_.count_include_pad.value()) {
		mode_ = miopenPoolingAverage;
        } else { 
           mode_= miopenPoolingAverageInclusive;
         }
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  ~CuDNNPoolingOp() {
    MIOPEN_CALL(miopenDestroyTensorDescriptor(in_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(out_desc_));
    MIOPEN_CALL(miopenDestroyPoolingDescriptor(pooling_desc_));
    if(workspace !=nullptr)
        hipFree(workspace);

  }

  void Forward(const OpContext &ctx, const TBlob &in_data,
      const OpReqType &req, const TBlob &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    this->Init(s, in_data, out_data);
    if (param_.kernel.ndim() == 2) {
      // 2d pool
      Tensor<gpu, 4, DType> data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);

      size_t temp_workspaceSize = 0;
      MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(out_desc_, &temp_workspaceSize));
      if (temp_workspaceSize > 0 && (temp_workspaceSize > workspaceSize || workspace == nullptr)) {
            workspaceSize = temp_workspaceSize;
            hipFree(workspace);
            hipMalloc(&workspace, workspaceSize);
          }


      MIOPEN_CALL(miopenPoolingForward(s->dnn_handle_,
                                     pooling_desc_,
                                     &alpha,
                                     in_desc_,
                                     data.dptr_,
                                     &beta,
                                     out_desc_,
                                     out.dptr_,
                                     true,
                                     workspace,
                                     workspaceSize));
    } else if (param_.kernel.ndim() == 3) {
      // 3d pool
      Tensor<gpu, 5, DType> data = in_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      size_t temp_workspaceSize = 0;
          MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(out_desc_, &temp_workspaceSize));
          if (temp_workspaceSize > 0 && (temp_workspaceSize > workspaceSize  || workspace == nullptr)) {
               workspaceSize = temp_workspaceSize;
               hipFree(workspace);
               hipMalloc(&workspace, workspaceSize);
          }

       MIOPEN_CALL(miopenPoolingForward(s->dnn_handle_,
                                     pooling_desc_,
                                     &alpha,
                                     in_desc_,
                                     data.dptr_,
                                     &beta,
                                     out_desc_,
                                     out.dptr_,
                                     true,
                                     workspace,
                                     workspaceSize));
    } else {
      LOG(FATAL) << "Only support 2D or 3D pooling";
    }
  }

  void Backward(const OpContext &ctx, const TBlob &out_grad,
      const TBlob &in_data, const TBlob &out_data,
      const OpReqType &req, const TBlob &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;

    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    this->Init(s, in_data, out_data);
    size_t temp_workspaceSize = 0;
      MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(out_desc_, &temp_workspaceSize));
      if (temp_workspaceSize > 0 && (temp_workspaceSize > workspaceSize || workspace == nullptr)) {
            workspaceSize = temp_workspaceSize;
            hipFree(workspace);
            hipMalloc(&workspace, workspaceSize);
          }
    if (param_.kernel.ndim() == 2) {
      // 2d pool
      Tensor<gpu, 4, DType> m_out_grad = out_grad.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_in_data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_out_data = out_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_in_grad = in_grad.get<gpu, 4, DType>(s);
      MIOPEN_CALL(miopenPoolingBackward(s->dnn_handle_,
                                      pooling_desc_,
                                      &alpha,
                                      out_desc_,
                                      m_out_data.dptr_,
                                      out_desc_,
                                      m_out_grad.dptr_,
                                      in_desc_,
                                      m_in_data.dptr_,
                                      &beta,
                                      in_desc_,
                                      m_in_grad.dptr_,
                                      workspace));

    } else if (param_.kernel.ndim() == 3) {
      // 3d pool
      Tensor<gpu, 5, DType> m_out_grad = out_grad.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_in_data = in_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_out_data = out_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_in_grad = in_grad.get<gpu, 5, DType>(s);

      MIOPEN_CALL(miopenPoolingBackward(s->dnn_handle_,
                                      pooling_desc_,
                                      &alpha,
                                      out_desc_,
                                      m_out_data.dptr_,
                                      out_desc_,
                                      m_out_grad.dptr_,
                                      in_desc_,
                                      m_in_data.dptr_,
                                      &beta,
                                      in_desc_,
                                      m_in_grad.dptr_,
                                      workspace));
    } else {
      LOG(FATAL) << "Only support 2D or 3D pooling";
    }
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s, const TBlob &in_data,
      const TBlob &out_data) {
    using namespace mshadow;
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      Tensor<gpu, 4, DType> data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data.get<gpu, 4, DType>(s);
      mshadow::Shape<4> dshape = data.shape_;
      MIOPEN_CALL(miopenSet4dTensorDescriptor(in_desc_,
                                            dtype_,
                                            data.shape_[0],
                                            data.shape_[1],
                                            data.shape_[2],
                                            data.shape_[3]));
      MIOPEN_CALL(miopenSet4dTensorDescriptor(out_desc_,
                                            dtype_,
                                            out.shape_[0],
                                            out.shape_[1],
                                            out.shape_[2],
                                            out.shape_[3]));


      MIOPEN_CALL(miopenSet2dPoolingDescriptor(pooling_desc_,
                                             mode_,
                                             param_.global_pool ? dshape[2] : param_.kernel[0],
                                             param_.global_pool ? dshape[3] : param_.kernel[1],
                                             param_.global_pool ? 0 : param_.pad[0],
                                             param_.global_pool ? 0 : param_.pad[1],
                                             param_.global_pool ? 1 : param_.stride[0],
                                             param_.global_pool ? 1 : param_.stride[1]));
    } else {
      Tensor<gpu, 5, DType> data = in_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data.get<gpu, 5, DType>(s);
      std::vector<int> ishape = {static_cast<int>(data.shape_[0]),
                                 static_cast<int>(data.shape_[1]),
                                 static_cast<int>(data.shape_[2]),
                                 static_cast<int>(data.shape_[3]),
                                 static_cast<int>(data.shape_[4])};

      std::vector<int> istride = {static_cast<int>(ishape[1] * ishape[2] * ishape[3] * ishape[4]),
                                  static_cast<int>(ishape[2] * ishape[3] * ishape[4]),
                                  static_cast<int>(ishape[3] * ishape[4]),
                                  static_cast<int>(ishape[4]), 1};

      std::vector<int> oshape = {static_cast<int>(out.shape_[0]),
                                 static_cast<int>(out.shape_[1]),
                                 static_cast<int>(out.shape_[2]),
                                 static_cast<int>(out.shape_[3]),
                                 static_cast<int>(out.shape_[4])};

      std::vector<int> ostride = {static_cast<int>(oshape[1] * oshape[2] * oshape[3] * oshape[4]),
                                  static_cast<int>(oshape[2] * oshape[3] * oshape[4]),
                                  static_cast<int>(oshape[3] * oshape[4]),
                                  static_cast<int>(oshape[4]), 1};

      std::vector<int> kernel_vec = {param_.global_pool ? ishape[2] :
                                                          static_cast<int>(param_.kernel[0]),
                                     param_.global_pool ? ishape[3] :
                                                          static_cast<int>(param_.kernel[1]),
                                     param_.global_pool ? ishape[4] :
                                                          static_cast<int>(param_.kernel[2])};

      std::vector<int> pad_vec = {param_.global_pool ? 0 : static_cast<int>(param_.pad[0]),
                                  param_.global_pool ? 0 : static_cast<int>(param_.pad[1]),
                                  param_.global_pool ? 0 : static_cast<int>(param_.pad[2])};

      std::vector<int> stride_vec = {param_.global_pool ? 1 : static_cast<int>(param_.stride[0]),
                                     param_.global_pool ? 1 : static_cast<int>(param_.stride[1]),
                                     param_.global_pool ? 1 : static_cast<int>(param_.stride[2])};

      int ndim = static_cast<int>(ishape.size());
      if (ndim > 0 && ndim < 6)
        MIOPEN_CALL(miopenSetTensorDescriptor(in_desc_,
                                              dtype_,
                                              static_cast<int>(ishape.size()),
                                              &ishape[0],
                                              &istride[0]));

       ndim = static_cast<int>(oshape.size());
      if (ndim > 0 && ndim < 6)
        MIOPEN_CALL(miopenSetTensorDescriptor(out_desc_,
                                              dtype_,
                                              static_cast<int>(oshape.size()),
                                              &oshape[0],
                                              &ostride[0]));

    }
     workspaceSize = 0;
   MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(out_desc_, &workspaceSize));
  if(workspaceSize > 0)
     if(workspace !=nullptr)
        hipFree(workspace);
   hipMalloc(&workspace, workspaceSize);
  }

  miopenDataType_t dtype_;
  miopenHandle_t handle_;
  miopenPoolingMode_t mode_;
  miopenTensorDescriptor_t in_desc_;
  miopenTensorDescriptor_t out_desc_;
  miopenPoolingDescriptor_t pooling_desc_;
  PoolingParam param_;

  size_t workspaceSize;
  void* workspace;
};  // class CuDNNPoolingOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_

