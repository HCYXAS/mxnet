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

#define USE_CUDNN_LSTM_PROJ MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7200

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
    input_mode_ = miopenRNNlinear; //CUDNN_LINEAR_INPUT;  // not supported in MIOpen
    // RNN Mode
    switch (param_.mode) {
      case rnn_enum::kRnnRelu:
        mode_ = miopenRNNRELU;
        break;
      case rnn_enum::kRnnTanh:
        mode_ = miopenRNNTANH;
        break;
      case rnn_enum::kLstm:
        mode_ = miopenLSTM;
        break;
      case rnn_enum::kGru:
        mode_ = miopenGRU;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
#if USE_CUDNN_LSTM_PROJ
    if (param_.projection_size.has_value()) {
      CHECK_EQ(param_.mode, rnn_enum::kLstm)
        << "Projection is only supported for LSTM.";
      CHECK_GE(param_.state_size, param_.projection_size.value())
        << "State size must be larger than projection size.";
    }
#else
    CHECK(!param_.projection_size.has_value())
      << "Projection is only supported for LSTM with CuDNN version later than 7.1.1.";
#endif
#if USE_CUDNN_LSTM_PROJ
    if (param_.lstm_state_clip_min.has_value()
        || param_.lstm_state_clip_max.has_value()) {
      CHECK_EQ(param_.mode, rnn_enum::kLstm)
        << "State clipping is only supported for LSTM.";
      CHECK(param_.lstm_state_clip_min.has_value() && param_.lstm_state_clip_max.has_value())
        << "lstm_state_clip_min and lstm_state_clip_max must be specified together.";
      CHECK_GE(param_.lstm_state_clip_max.value(), param_.lstm_state_clip_min.value())
        << "lstm_state_clip_max must be greater or equal to lstm_state_clip_min";
    }
#else
    CHECK(!param_.lstm_state_clip_min.has_value()
          && !param_.lstm_state_clip_max.has_value())
      << "State clipping is only supported for LSTM with CuDNN version later than 7.2.1.";
#endif
    // RNN Direction
    direction_ = param_.bidirectional ? miopenRNNbidirection: miopenRNNunidirection;
    // Other
    if (param_.mode == rnn_enum::kLstm)
      param_.lstm_q_ = true;
    else
      param_.lstm_q_ = false;

    // Create descriptors
    CUDNN_CALL(miopenCreateTensorDescriptor(&hx_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&cx_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&hy_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&cy_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&dhx_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&dcx_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&dhy_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&dcy_desc_));

    CUDNN_CALL(miopenCreateTensorDescriptor(&w_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&dw_desc_));

    CUDNN_CALL(miopenCreateRNNDescriptor(&rnn_desc_)); 
   /*CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_)); */ //TODO cudnnCreateDropoutDescriptor not supported in MIOpen

   /* #if USE_CUDNN_LSTM_PROJ
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&x_data_desc_)); 
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&y_data_desc_)); 
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&dx_data_desc_)); 
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&dy_data_desc_)); /
    #endif */ //TODO cudnnCreateRNNDataDescriptor not supported in MIOpen
  }

  ~CuDNNRNNOp() {
    CUDNN_CALL(miopenDestroyTensorDescriptor(hx_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(cx_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(hy_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(cy_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(dhx_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(dcx_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(dhy_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(dcy_desc_));
//using Tensor as no equivalent available for Filter (2APIs)
    CUDNN_CALL(miopenDestroyTensorDescriptor(w_desc_));
    CUDNN_CALL(miopenDestroyTensorDescriptor(dw_desc_));
    CUDNN_CALL(miopenDestroyRNNDescriptor(rnn_desc_));
    //CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));

    if (init_cudnn_) {
      for (size_t i = 0; i < x_desc_vec_.size(); ++i) {
        CUDNN_CALL(miopenDestroyTensorDescriptor(x_desc_vec_[i]));
        CUDNN_CALL(miopenDestroyTensorDescriptor(y_desc_vec_[i]));
        CUDNN_CALL(miopenDestroyTensorDescriptor(dx_desc_vec_[i]));
        CUDNN_CALL(miopenDestroyTensorDescriptor(dy_desc_vec_[i]));
      }
      init_cudnn_ = false;

      Storage::Get()->Free(reserve_space_);
      if (param_.p > 0) {
        Storage::Get()->Free(dropout_states_);
      }
    }
   /* #if USE_CUDNN_LSTM_PROJ
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(x_data_desc_));
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(y_data_desc_));
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(dx_data_desc_));
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(dy_data_desc_));
    #endif */ //TODO cudnnDestroyRNNDataDescriptor not supported in MIOpen
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
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
   /* #if USE_CUDNN_LSTM_PROJ
    std::vector<int> seqLengthArray(param_.batch_size_, param_.seq_length_);
    CUDNN_CALL(cudnnSetRNNDataDescriptor(x_data_desc_,
                                         dtype_,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                         param_.seq_length_,
                                         param_.batch_size_,
                                         param_.input_size_,
                                         seqLengthArray.data(),
                                         nullptr));
    int out_size =
      (param_.projection_size.has_value()) ? param_.projection_size.value() : param_.state_size;
    out_size = (param_.bidirectional) ? (out_size * 2) : out_size;
    CUDNN_CALL(cudnnSetRNNDataDescriptor(y_data_desc_,
                                         dtype_,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                         param_.seq_length_,
                                         param_.batch_size_,
                                         out_size,
                                         seqLengthArray.data(),
                                         nullptr));
    if (ctx.is_train) {
      CUDNN_CALL(cudnnSetRNNDataDescriptor(dx_data_desc_,
                                           dtype_,
                                           CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                           param_.seq_length_,
                                           param_.batch_size_,
                                           param_.input_size_,
                                           seqLengthArray.data(),
                                           nullptr));
      CUDNN_CALL(cudnnSetRNNDataDescriptor(dy_data_desc_,
                                           dtype_,
                                           CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                           param_.seq_length_,
                                           param_.batch_size_,
                                           out_size,
                                           seqLengthArray.data(),
                                           nullptr));
    }
    #endif */  //TODO cudnnSetRNNDataDescriptor not supported in MIOpen

    #if USE_CUDNN_LSTM_PROJ
    bool clip_state = param_.lstm_state_clip_min.has_value();
    bool clip_nan = param_.lstm_state_clip_nan;
    /* CUDNN_CALL(cudnnRNNSetClip(s->dnn_handle_,
                               rnn_desc_,
                               clip_state ? CUDNN_RNN_CLIP_MINMAX : CUDNN_RNN_CLIP_NONE,
                               clip_nan ? CUDNN_NOT_PROPAGATE_NAN : CUDNN_PROPAGATE_NAN,
                               clip_state ? param_.lstm_state_clip_min.value() : 0.0,
                               clip_state ? param_.lstm_state_clip_max.value() : 0.0)); */ //TODO cudnnRNNSetClip not supported in MIOpen
    #endif

    if (ctx.is_train) {
      #if USE_CUDNN_LSTM_PROJ
      /*CUDNN_CALL(cudnnRNNForwardTrainingEx(s->dnn_handle_,
                                           rnn_desc_,
                                           x_data_desc_,
                                           x.dptr_,
                                           hx_desc_,
                                           hx.dptr_,
                                           cx_desc_,
                                           cx_ptr,
                                           w_desc_,
                                           w.dptr_,
                                           y_data_desc_,
                                           y.dptr_,
                                           hy_desc_,
                                           hy_ptr,
                                           cy_desc_,
                                           cy_ptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           temp_space.dptr_,
                                           workspace_byte_,
                                           reserve_space_.dptr,
                                           reserve_space_byte_)); */ //TODO as cudnnRNNForwardTrainingEx not supported in MIOpen
      #else
      CUDNN_CALL(miopenRNNForwardTraining(s->dnn_handle_,
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
      #endif
    } else {
      #if USE_CUDNN_LSTM_PROJ
      /*CUDNN_CALL(cudnnRNNForwardInferenceEx(s->dnn_handle_,
                                            rnn_desc_,
                                            x_data_desc_,
                                            x.dptr_,
                                            hx_desc_,
                                            hx.dptr_,
                                            cx_desc_,
                                            cx_ptr,
                                            w_desc_,
                                            w.dptr_,
                                            y_data_desc_,
                                            y.dptr_,
                                            hy_desc_,
                                            hy_ptr,
                                            cy_desc_,
                                            cy_ptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            temp_space.dptr_,
                                            workspace_byte_));*/ //TODO as cudnnRNNForwardInferenceEx is not supported
      #else
      CUDNN_CALL(miopenRNNForwardInference(s->dnn_handle_,
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
      #endif
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
    #if USE_CUDNN_LSTM_PROJ
   /* CUDNN_CALL(cudnnRNNBackwardDataEx(s->dnn_handle_,
                                      rnn_desc_,
                                      y_data_desc_,
                                      y.dptr_,
                                      dy_data_desc_,
                                      dy.dptr_,
                                      nullptr,
                                      nullptr,
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
                                      dx_data_desc_,
                                      dx.dptr_,
                                      dhx_desc_,
                                      dhx.dptr_,
                                      dcx_desc_,
                                      dcx_ptr,
                                      nullptr,
                                      nullptr,
                                      temp_space.dptr_,
                                      workspace_byte_,
                                      reserve_space_.dptr,
                                      reserve_space_byte_)); */ //TODO as cudnnRNNBackwardDataEx not supported in MIOpen
   /* CUDNN_CALL(cudnnRNNBackwardWeightsEx(s->dnn_handle_,
                                         rnn_desc_,
                                         x_data_desc_,
                                         x.dptr_,
                                         hx_desc_,
                                         hx.dptr_,
                                         y_data_desc_,
                                         y.dptr_,
                                         temp_space.dptr_,
                                         workspace_byte_,
                                         dw_desc_,
                                         dw.dptr_,
                                         reserve_space_.dptr,
                                         reserve_space_byte_)); */ //TODO as cudnnRNNBackwardWeightsEx not supported in MIOPen 
    #else
    CUDNN_CALL(miopenRNNBackwardData(s->dnn_handle_,
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
/*    CUDNN_CALL(cudnnRNNBackwardWeights(s->dnn_handle_,
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
                                       reserve_space_byte_));*/

   CUDNN_CALL(miopenRNNBackwardWeights(s->dnn_handle_,
                                       rnn_desc_,
                                       param_.seq_length_,
                                       x_desc_vec_.data(),
                                       x.dptr_,
                                       hx_desc_,
                                       hx.dptr_,
                                       y_desc_vec_.data(),
                                       y.dptr_,
                                       dw_desc_,
                                       dw.dptr_,
                                       temp_space.dptr_,
                                       workspace_byte_,
                                       reserve_space_.dptr,
                                       reserve_space_byte_));
 #endif
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR >= 5
    //format_ = CUDNN_TENSOR_NCHW;
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
      std::vector<miopenTensorDescriptor_t> x_vec(param_.seq_length_);
      std::vector<miopenTensorDescriptor_t> y_vec(param_.seq_length_);
      std::vector<miopenTensorDescriptor_t> dx_vec(param_.seq_length_);
      std::vector<miopenTensorDescriptor_t> dy_vec(param_.seq_length_);
      int dimA[3];
      int strideA[3];
      for (int i = 0; i < param_.seq_length_; i++) {
        CUDNN_CALL(miopenCreateTensorDescriptor(&x_vec[i]));
        CUDNN_CALL(miopenCreateTensorDescriptor(&y_vec[i]));
        CUDNN_CALL(miopenCreateTensorDescriptor(&dx_vec[i]));
        CUDNN_CALL(miopenCreateTensorDescriptor(&dy_vec[i]));

        dimA[0] = param_.batch_size_;
        dimA[1] = param_.input_size_;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(miopenSetTensorDescriptor(x_vec[i],
                                              dtype_,  // TODO Currently only miopenFloat is implemented
                                              3,
                                              dimA,
                                              strideA));
        CUDNN_CALL(miopenSetTensorDescriptor(dx_vec[i],
                                              dtype_, // TODO Currently only miopenFloat is implemented
                                              3,
                                              dimA,
                                              strideA));
        dimA[0] = param_.batch_size_;
        dimA[1] = param_.bidirectional ? param_.state_size * 2 : param_.state_size;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(miopenSetTensorDescriptor(y_vec[i],
                                             dtype_, // TODO Currently only miopenFloat is implemented
                                             3,
                                             dimA,
                                             strideA));
        CUDNN_CALL(miopenSetTensorDescriptor(dy_vec[i],
                                              dtype_, // TODO Currently only miopenFloat is implemented
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
      #if USE_CUDNN_LSTM_PROJ
      int dimB[3];
      int strideB[3];
      dimB[0] = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimB[1] = param_.batch_size_;
      dimB[2] = param_.projection_size.has_value() ?
                param_.projection_size.value() : param_.state_size;
      strideB[0] = dimB[2] * dimB[1];
      strideB[1] = dimB[2];
      strideB[2] = 1;
      #endif

      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(miopeSetTensorDescriptor(hx_desc_,
                                            dtype_,// TODO Currently only miopenFloat is implemented
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(miopenSetTensorDescriptor(hx_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(miopenSetTensorDescriptor(cx_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(miopenSetTensorDescriptor(hy_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(miopenSetTensorDescriptor(hy_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(miopenSetTensorDescriptor(cy_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(miopenSetTensorDescriptor(dhx_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(miopenSetTensorDescriptor(dhx_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(miopenSetTensorDescriptor(dcx_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(miopenSetTensorDescriptor(dhy_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(miopenSetTensorDescriptor(dhy_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(miopenSetTensorDescriptor(dcy_desc_,
                                            dtype_, // TODO Currently only miopenFloat is implemented
                                            3,
                                            dimA,
                                            strideA));

      // Create Dropout descriptors
      /*if (param_.p > 0) {
        CUDNN_CALL(cudnnDropoutGetStatesSize(s->dnn_handle_, &dropout_byte_));
        dropout_size_ = dropout_byte_ / sizeof(DType);
        dropout_states_ = Storage::Get()->Alloc(dropout_byte_, Context::GPU());
      } else {
        dropout_states_ = {};
        dropout_byte_ = 0;
      }
      CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc_, s->dnn_handle_,
                                           param_.p,  // discard probability
                                           dropout_states_.dptr, dropout_byte_,
                                           seed_));*/ //TODO MIOpen does not support Dropout
      // RNN descriptors
      /*#if CUDNN_MAJOR >= 6
      cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;
      CUDNN_CALL(cudnnSetRNNDescriptor_v6(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.state_size,
                                          param_.num_layers,
                                          dropout_desc_,
                                          input_mode_,
                                          direction_,
                                          mode_,
                                          rnn_algo,
                                          dtype_));
      #else*/// TODO commented as not supported in MIOpen
      CUDNN_CALL(miopenSetRNNDescriptor(rnn_desc_,
                                       param_.state_size,
                                       param_.num_layers,
                                       input_mode_,
                                       direction_,
                                       mode_,
                                       miopenRNNwithBias,
                                       miopenRNNdefault,
                                       dtype_));
      //#endif
      /*#if CUDNN_MAJOR >= 7
        cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
        if (cudnn_tensor_core_ && rnn_algo == CUDNN_RNN_ALGO_STANDARD) {
          math_type = CUDNN_TENSOR_OP_MATH;
        }
      #if CUDNN_VERSION >= 7200
            if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
                (DataType<DType>::kFlag != kFloat16))
              math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
      #endif
        CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_, math_type));
      #endif*/ //TODO commented as unsupported in MIOpen
      #if USE_CUDNN_LSTM_PROJ
      if (param_.projection_size.has_value()) {
        CUDNN_CALL(cudnnSetRNNProjectionLayers(s->dnn_handle_,
                                               rnn_desc_,
                                               param_.projection_size.value(),
                                               0));
      }
      #endif
      // Get temp space sizes
      CUDNN_CALL(miopenGetRNNWorkspaceSize(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.seq_length_,
                                          x_desc_vec_.data(),
                                          &workspace_byte_));
      CUDNN_CALL(miopenGetRNNTrainingReserveSize(s->dnn_handle_,
                                                rnn_desc_,
                                                param_.seq_length_,
                                                x_desc_vec_.data(),
                                                &reserve_space_byte_));
      workspace_size_ = workspace_byte_ / sizeof(DType);
      // Allocate the reserve space
      reserve_space_ = Storage::Get()->Alloc(reserve_space_byte_, Context::GPU());

      // Check that number of params are correct
      size_t cudnn_param_size;
       CUDNN_CALL(miopenGetRNNParamsSize(s->dnn_handle_,
                                       rnn_desc_,
                                       x_desc_vec_[0],
                                       &cudnn_param_size,
                                       dtype_));
      CHECK_EQ(w.shape_[0] * sizeof(DType), cudnn_param_size);

      // Set param descriptors
      int dim_w[3] = {1, 1, 1};
      dim_w[0] = w.shape_[0];
//for stride error
      int stride_w[3];
      stride_w[0] = dim_w[2] * dim_w[1];
      stride_w[1] = dim_w[2];
      stride_w[2] = 1;
      CUDNN_CALL(miopenSetTensorDescriptor(w_desc_,
                                            dtype_,
                                            //format_,
                                            3,
                                            dim_w,
                                            stride_w));
      CUDNN_CALL(miopenSetTensorDescriptor(dw_desc_,
                                            dtype_,
                                            //format_,
                                            3,
                                            dim_w,
                                            stride_w));

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

  miopenDataType_t dtype_;
  bool init_cudnn_;
  miopenRNNDescriptor_t rnn_desc_;
  miopenRNNMode_t mode_;
  miopenRNNDirectionMode_t direction_;
  miopenRNNInputMode_t input_mode_;
  //cudnnDropoutDescriptor_t dropout_desc_; //TODO commented as unsupported in MIOpen
  Storage::Handle dropout_states_, reserve_space_;
  uint64_t seed_ = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
  size_t workspace_byte_, reserve_space_byte_, dropout_byte_;
  int workspace_size_, dropout_size_;
  std::vector<miopenTensorDescriptor_t> x_desc_vec_, y_desc_vec_, dx_desc_vec_, dy_desc_vec_;
  #if USE_CUDNN_LSTM_PROJ
  cudnnRNNDataDescriptor_t x_data_desc_, y_data_desc_, dx_data_desc_, dy_data_desc_; //check for equivalent
  #endif
  miopenTensorDescriptor_t hx_desc_, cx_desc_;
  miopenTensorDescriptor_t hy_desc_, cy_desc_;
  miopenTensorDescriptor_t dhx_desc_, dcx_desc_;
  miopenTensorDescriptor_t dhy_desc_, dcy_desc_;

  miopenTensorDescriptor_t w_desc_, dw_desc_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;

  /*#if CUDNN_MAJOR >= 5
  cudnnTensorFormat_t format_;
  #endif*/
  RNNParam param_;
};
#endif  // __HIPCC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_RNN_INL_H_
