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
 * Copyright (c) 2016 by Contributors
 * \file cudnn_rnn-inl.h
 * \brief
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_CUDNN_RNN_INL_H_
#define MXNET_OPERATOR_CUDNN_RNN_INL_H_

#include <mxnet/storage.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cstdint>
#include "./rnn-inl.h"

namespace mxnet {
namespace op {
#if defined(__HIPCC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
template<typename DType>
class CuDNNRNNOp : public Operator {
 public:
  explicit CuDNNRNNOp(RNNParam param) {
    this->param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    // TensorCore algos only allowed on fp16-I/O convolutions if permitted by the global policy.
    // No tests in place for fp16 RNNs, so leave TensorCore disabled for now.
    cudnn_tensor_core_ = false;
    // When fp16 RNN tests are introduced, we can enable TensorCore as follows:
//    cudnn_tensor_core =
//        mshadow::DataType<DType>::kFlag == mshadow::kFloat16 && GetEnvAllowTensorCore();
    // Defaults
    input_mode_ = HIPDNN_LINEAR_INPUT;  // Don't support this yet
    // RNN Mode
    switch (param_.mode) {
      case rnn_enum::kRnnRelu:
        mode_ = HIPDNN_RNN_RELU;
        break;
      case rnn_enum::kRnnTanh:
        mode_ = HIPDNN_RNN_TANH;
        break;
      case rnn_enum::kLstm:
        mode_ = HIPDNN_LSTM;
        break;
      case rnn_enum::kGru:
        mode_ = HIPDNN_GRU;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
    // RNN Direction
    direction_ = param_.bidirectional ? HIPDNN_BIDIRECTIONAL : HIPDNN_UNIDIRECTIONAL;
    // Other
    if (param_.mode == rnn_enum::kLstm)
      param_.lstm_q_ = true;
    else
      param_.lstm_q_ = false;
  }

  ~CuDNNRNNOp() {
    if (init_cudnn_) {
      for (size_t i = 0; i < x_desc_vec_.size(); ++i) {
        CUDNN_CALL(hipdnnDestroyTensorDescriptor(x_desc_vec_[i]));
        CUDNN_CALL(hipdnnDestroyTensorDescriptor(y_desc_vec_[i]));
        CUDNN_CALL(hipdnnDestroyTensorDescriptor(dx_desc_vec_[i]));
        CUDNN_CALL(hipdnnDestroyTensorDescriptor(dy_desc_vec_[i]));
      }
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(hx_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(cx_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(hy_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(cy_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(dhx_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(dcx_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(dhy_desc_));
      CUDNN_CALL(hipdnnDestroyTensorDescriptor(dcy_desc_));

      CUDNN_CALL(hipdnnDestroyFilterDescriptor(w_desc_));
      CUDNN_CALL(hipdnnDestroyRNNDescriptor(rnn_desc_));
      CUDNN_CALL(hipdnnDestroyDropoutDescriptor(dropout_desc_));
      Storage::Get()->Free(dropout_states_);
      Storage::Get()->Free(reserve_space_);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs)
        out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // get input + output tensors
    Tensor<gpu, 3, DType> x = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 1, DType> w = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 3, DType> hx = in_data[rnn_enum::kState].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> y = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);

    void * hy_ptr = NULL;
    if (param_.state_outputs)
      hy_ptr = out_data[rnn_enum::kStateOut].get<gpu, 3, DType>(s).dptr_;

    DType * cx_ptr = NULL;
    DType * cy_ptr = NULL;

    if (param_.lstm_q_)
      cx_ptr = (in_data[rnn_enum::kStateCell].get<gpu, 3, DType>(s)).dptr_;
    if (param_.lstm_q_ && param_.state_outputs)
      cy_ptr = (out_data[rnn_enum::kStateCellOut].get<gpu, 3, DType>(s)).dptr_;

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);

    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }
    // Get temp space
    int temp_size = workspace_size_;
    Tensor<gpu, 1, DType> temp_space =
      ctx.requested[rnn_enum::kTempSpace].get_space_typed<gpu, 1, DType>(
                              mshadow::Shape1(temp_size), s);
    if (ctx.is_train) {
      CUDNN_CALL(hipdnnRNNForwardTraining(s->dnn_handle_,
                                         rnn_desc_,
                                         param_.seq_length_,
                                         x_desc_vec_.data(),
                                         x.dptr_,
                                         hx_desc_,
                                         hx.dptr_,
                                         cx_desc_,
                                         cx_ptr,
                                         w_desc_,
                                         w.dptr_,
                                         y_desc_vec_.data(),
                                         y.dptr_,
                                         hy_desc_,
                                         hy_ptr,
                                         cy_desc_,
                                         cy_ptr,
                                         temp_space.dptr_,
                                         workspace_byte_,
                                         reserve_space_.dptr,
                                         reserve_space_byte_));
    } else {
      // inference mode
      CUDNN_CALL(hipdnnRNNForwardInference(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.seq_length_,
                                          x_desc_vec_.data(),
                                          x.dptr_,
                                          hx_desc_,
                                          hx.dptr_,
                                          cx_desc_,
                                          cx_ptr,
                                          w_desc_,
                                          w.dptr_,
                                          y_desc_vec_.data(),
                                          y.dptr_,
                                          hy_desc_,
                                          hy_ptr,
                                          cy_desc_,
                                          cy_ptr,
                                          temp_space.dptr_,
                                          workspace_byte_));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs)
      out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(out_grad.size(), out_expected);
    CHECK_EQ(req.size(), in_expected);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo) << "AddTo is not supported for state";
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // get input + output tensors
    Tensor<gpu, 3, DType> x = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> dx = in_grad[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 1, DType> w = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 1, DType> dw = in_grad[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 3, DType> hx = in_data[rnn_enum::kState].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> dhx = in_grad[rnn_enum::kState].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> y = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> dy = out_grad[rnn_enum::kOut].get<gpu, 3, DType>(s);
    if (req[rnn_enum::kParams] != kAddTo) {
      dw = mshadow::expr::ScalarExp<DType>(0.0f);
    }
    // only need kStateOut grad output_states is true
    void * dhy_ptr = NULL;
    if (param_.state_outputs)
      dhy_ptr = out_grad[rnn_enum::kStateOut].get<gpu, 3, DType>(s).dptr_;

    // Deal with lstm
    void * dcx_ptr = NULL;
    void * dcy_ptr = NULL;
    void * cx_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo) << "AddTo is not supported for state cell";
      cx_ptr = (in_data[rnn_enum::kStateCell].get<gpu, 3, DType>(s)).dptr_;
      dcx_ptr = (in_grad[rnn_enum::kStateCell].get<gpu, 3, DType>(s)).dptr_;
    }
    if ((param_.mode == rnn_enum::kLstm) && param_.state_outputs)
        dcy_ptr = (out_grad[rnn_enum::kStateCellOut].get<gpu, 3, DType>(s)).dptr_;

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(dw.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(dhx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);
    CHECK_EQ(dy.CheckContiguous(), true);

    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }

    // Get temp space
    int temp_size = workspace_size_;
    Tensor<gpu, 1, DType> temp_space =
      ctx.requested[rnn_enum::kTempSpace].get_space_typed<gpu, 1, DType>(
                              mshadow::Shape1(temp_size), s);
    CUDNN_CALL(hipdnnRNNBackwardData(s->dnn_handle_,
                                    rnn_desc_,
                                    param_.seq_length_,
                                    y_desc_vec_.data(),
                                    y.dptr_,
                                    dy_desc_vec_.data(),
                                    dy.dptr_,
                                    dhy_desc_,
                                    dhy_ptr,
                                    dcy_desc_,
                                    dcy_ptr,
                                    w_desc_,
                                    w.dptr_,
                                    hx_desc_,
                                    hx.dptr_,
                                    cx_desc_,
                                    cx_ptr,
                                    dx_desc_vec_.data(),
                                    dx.dptr_,
                                    dhx_desc_,
                                    dhx.dptr_,
                                    dcx_desc_,
                                    dcx_ptr,
                                    temp_space.dptr_,
                                    workspace_byte_,
                                    reserve_space_.dptr,
                                    reserve_space_byte_));
    CUDNN_CALL(hipdnnRNNBackwardWeights(s->dnn_handle_,
                                       rnn_desc_,
                                       param_.seq_length_,
                                       x_desc_vec_.data(),
                                       x.dptr_,
                                       hx_desc_,
                                       hx.dptr_,
                                       y_desc_vec_.data(),
                                       y.dptr_,
                                       temp_space.dptr_,
                                       workspace_byte_,
                                       dw_desc_,
                                       dw.dptr_,
                                       reserve_space_.dptr,
                                       reserve_space_byte_));
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR >= 5
    format_ = HIPDNN_TENSOR_NCHW;
    #endif
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs)
      out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      // get input + output tensors
      Tensor<gpu, 3, DType> x = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
      Tensor<gpu, 1, DType> w = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
      param_.seq_length_ = x.shape_[0];
      param_.batch_size_ = x.shape_[1];
      param_.input_size_ = x.shape_[2];

      // Tensor Descriptors
      std::vector<hipdnnTensorDescriptor_t> x_vec(param_.seq_length_);
      std::vector<hipdnnTensorDescriptor_t> y_vec(param_.seq_length_);
      std::vector<hipdnnTensorDescriptor_t> dx_vec(param_.seq_length_);
      std::vector<hipdnnTensorDescriptor_t> dy_vec(param_.seq_length_);
      int dimA[3];
      int strideA[3];
      for (int i = 0; i < param_.seq_length_; i++) {
        CUDNN_CALL(hipdnnCreateTensorDescriptor(&x_vec[i]));
        CUDNN_CALL(hipdnnCreateTensorDescriptor(&y_vec[i]));
        CUDNN_CALL(hipdnnCreateTensorDescriptor(&dx_vec[i]));
        CUDNN_CALL(hipdnnCreateTensorDescriptor(&dy_vec[i]));

        dimA[0] = param_.batch_size_;
        dimA[1] = param_.input_size_;
        dimA[2] = 1;
        dimA[0] = param_.batch_size_;
        dimA[1] = param_.input_size_;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(hipdnnSetTensorNdDescriptor(x_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
        CUDNN_CALL(hipdnnSetTensorNdDescriptor(dx_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
        dimA[0] = param_.batch_size_;
        dimA[1] = param_.bidirectional ? param_.state_size * 2 : param_.state_size;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(hipdnnSetTensorNdDescriptor(y_vec[i],
                                             dtype_,
                                             3,
                                             dimA,
                                             strideA));
        CUDNN_CALL(hipdnnSetTensorNdDescriptor(dy_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
      }
      x_desc_vec_ = x_vec;
      y_desc_vec_ = y_vec;
      dx_desc_vec_ = dx_vec;
      dy_desc_vec_ = dy_vec;

      // set the state tensors
      dimA[0] = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimA[1] = param_.batch_size_;
      dimA[2] = param_.state_size;
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      CUDNN_CALL(hipdnnCreateTensorDescriptor(&hx_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&cx_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&hy_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&cy_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&dhx_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&dcx_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&dhy_desc_));
      CUDNN_CALL(hipdnnCreateTensorDescriptor(&dcy_desc_));

      CUDNN_CALL(hipdnnSetTensorNdDescriptor(hx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(cx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(hy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(cy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(dhx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(dcx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(dhy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      CUDNN_CALL(hipdnnSetTensorNdDescriptor(dcy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));

      // Create Dropout descriptors
      CUDNN_CALL(hipdnnCreateDropoutDescriptor(&dropout_desc_));
      CUDNN_CALL(hipdnnDropoutGetStatesSize(s->dnn_handle_,
                                           &dropout_byte_));
      dropout_size_ = dropout_byte_ / sizeof(DType);
      dropout_states_ = Storage::Get()->Alloc(dropout_byte_, Context::GPU());
      CUDNN_CALL(hipdnnSetDropoutDescriptor(dropout_desc_,
                                           s->dnn_handle_,
                                           param_.p,  // keep probability
                                           dropout_states_.dptr,
                                           dropout_byte_,
                                           seed_));
      // RNN descriptors
      CUDNN_CALL(hipdnnCreateRNNDescriptor(&rnn_desc_));

      #if CUDNN_MAJOR >= 6
        hipdnnRNNAlgo_t rnn_algo = HIPDNN_RNN_ALGO_STANDARD;
        CUDNN_CALL(hipdnnSetRNNDescriptor_v6(s->dnn_handle_,
                                            rnn_desc_,
                                            param_.state_size,
                                            param_.num_layers,
                                            dropout_desc_,
                                            input_mode_,
                                            direction_,
                                            mode_,
                                            rnn_algo,
                                            dtype_));
      #else
        CUDNN_CALL(hipdnnSetRNNDescriptor(rnn_desc_,
                                         param_.state_size,
                                         param_.num_layers,
                                         dropout_desc_,
                                         input_mode_,
                                         direction_,
                                         mode_,
                                         dtype_));
      #endif
      #if CUDNN_MAJOR >= 7
        hipdnnMathType_t math_type = HIPDNN_DEFAULT_MATH;
        if (cudnn_tensor_core_ && rnn_algo == HIPDNN_RNN_ALGO_STANDARD) {
          math_type = HIPDNN_TENSOR_OP_MATH;
        }
        CUDNN_CALL(hipdnnSetRNNMatrixMathType(rnn_desc_, math_type));
      #endif
      // Get temp space sizes
      CUDNN_CALL(hipdnnGetRNNWorkspaceSize(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.seq_length_,
                                          x_desc_vec_.data(),
                                          &workspace_byte_));
      CUDNN_CALL(hipdnnGetRNNTrainingReserveSize(s->dnn_handle_,
                                                rnn_desc_,
                                                param_.seq_length_,
                                                x_desc_vec_.data(),
                                                &reserve_space_byte_));
      workspace_size_ = workspace_byte_ / sizeof(DType);
      // Allocate the reserve space
      reserve_space_ = Storage::Get()->Alloc(reserve_space_byte_, Context::GPU());

      // Check that number of params are correct
      size_t cudnn_param_size;
      CUDNN_CALL(hipdnnGetRNNParamsSize(s->dnn_handle_,
                                       rnn_desc_,
                                       x_desc_vec_[0],
                                       &cudnn_param_size,
                                       dtype_));
      CHECK_EQ(w.shape_[0] * sizeof(DType), cudnn_param_size);

      // Set param descriptors
      CUDNN_CALL(hipdnnCreateFilterDescriptor(&w_desc_));
      CUDNN_CALL(hipdnnCreateFilterDescriptor(&dw_desc_));
      int dim_w[3] = {1, 1, 1};
      dim_w[0] = w.shape_[0];
      CUDNN_CALL(hipdnnSetFilterNdDescriptor(w_desc_,
                                            dtype_,
                                            format_,
                                            3,
                                            dim_w));
      CUDNN_CALL(hipdnnSetFilterNdDescriptor(dw_desc_,
                                            dtype_,
                                            format_,
                                            3,
                                            dim_w));

      // Query weight layout
      // cudnnFilterDescriptor_t m_desc;
      // CHECK_EQ(cudnnCreateFilterDescriptor(&m_desc), CUDNN_STATUS_SUCCESS);
      // DType *p;
      // int n = 2;
      // int64_t last = 0;
      // if (param_.mode == rnn_enum::kLstm) n = 8;
      // else if (param_.mode == rnn_enum::kGru) n = 6;

      // for (int i = 0; i < param_.num_layers*(param_.bidirectional?2:1); ++i) {
      //   for (int j = 0; j < n; ++j) {
      //     CHECK_EQ(cudnnGetRNNLinLayerMatrixParams(s->dnn_handle_, rnn_desc_,
      //       i, x_desc_vec_[0], w_desc_, 0, j, m_desc, (void**)&p), CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << ((int64_t)(p - NULL))/sizeof(DType) - last;
      //     last = ((int64_t)(p - NULL))/sizeof(DType);
      //     cudnnDataType_t t;
      //     cudnnTensorFormat_t f;
      //     int ndim = 5;
      //     int dims[5] = {0, 0, 0, 0, 0};
      //     CHECK_EQ(cudnnGetFilterNdDescriptor(m_desc, ndim, &t, &f, &ndim, &dims[0]),
      //       CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << "w: " <<  i << " " << j << " " << ((int64_t)(p - NULL))/sizeof(DType);
      //     for (int i = 0; i < ndim; ++i) LOG(INFO) << dims[i];
      //   }
      // }

      // for (int i = 0; i < param_.num_layers*(param_.bidirectional?2:1); ++i) {
      //   for (int j = 0; j < n; ++j) {
      //     CHECK_EQ(cudnnGetRNNLinLayerBiasParams(s->dnn_handle_, rnn_desc_, i, x_desc_vec_[0],
      //       w_desc_, 0, j, m_desc, (void**)&p), CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << ((int64_t)(p - NULL))/sizeof(DType) - last;
      //     last = ((int64_t)(p - NULL))/sizeof(DType);
      //     LOG(INFO) << "b: " << i << " " << j << " " << ((int64_t)(p - NULL))/sizeof(DType);
      //   }
      // }
    }
  }

  hipdnnDataType_t dtype_;
  bool init_cudnn_;
  hipdnnRNNDescriptor_t rnn_desc_;
  hipdnnRNNMode_t mode_;
  hipdnnDirectionMode_t direction_;
  hipdnnRNNInputMode_t input_mode_;
  hipdnnDropoutDescriptor_t dropout_desc_;
  Storage::Handle dropout_states_, reserve_space_;
  uint64_t seed_ = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
  size_t workspace_byte_, reserve_space_byte_, dropout_byte_;
  int workspace_size_, dropout_size_;
  std::vector<hipdnnTensorDescriptor_t> x_desc_vec_, y_desc_vec_, dx_desc_vec_, dy_desc_vec_;
  hipdnnTensorDescriptor_t hx_desc_, cx_desc_;
  hipdnnTensorDescriptor_t hy_desc_, cy_desc_;
  hipdnnTensorDescriptor_t dhx_desc_, dcx_desc_;
  hipdnnTensorDescriptor_t dhy_desc_, dcy_desc_;

  hipdnnFilterDescriptor_t w_desc_, dw_desc_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;

  #if CUDNN_MAJOR >= 5
  hipdnnTensorFormat_t format_;
  #endif
  RNNParam param_;
};
#endif  // __HIPCC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_RNN_INL_H_
