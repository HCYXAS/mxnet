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
 * \file cudnn_deconvolution-inl.h
 * \brief
 * \author Wei Wu, Leonard Lausen
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_

#include <mxnet/storage.h>
#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include "../deconvolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../../../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_MIOPEN == 1

template<typename DType>
class CuDNNDeconvolutionOp {
 public:
  CuDNNDeconvolutionOp() {
    MIOPEN_CALL(miopenCreateTensorDescriptor(&in_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&out_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&bias_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&filter_desc_));
    MIOPEN_CALL(miopenCreateConvolutionDescriptor(&forward_conv_desc_));
    MIOPEN_CALL(miopenCreateConvolutionDescriptor(&back_conv_desc_));
    MIOPEN_CALL(miopenCreateConvolutionDescriptor(&back_conv_desc_w_));
  }

  void Init(DeconvolutionParam param,
            int forward_compute_type,
            int backward_compute_type,
            const std::vector<TShape>& in_shape,
            const std::vector<TShape>& out_shape,
            const RunContext& rctx,
            bool add_to_weight) {
    using namespace mshadow;
    this->param_ = param;
    this->add_to_weight_ = add_to_weight;
    InitBufferForParam();
    auto cudnn_forward_compute_type = convertToCuDNNDataType(forward_compute_type);
    auto cudnn_backward_compute_type = convertToCuDNNDataType(backward_compute_type);
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    // TensorCore algos only allowed on fp16-I/O deconvolutions if permitted by the global policy.
    cudnn_tensor_core_ = DataType<DType>::kFlag == kFloat16 && GetEnvAllowTensorCore();

    auto effective_layout = param_.layout.value();
    switch (effective_layout) {
      // 1D convolutions will be executed as 2D convolutions with a height of 1.
      case mshadow::kNCW: effective_layout = mshadow::kNCHW; break;
      case mshadow::kNWC: effective_layout = mshadow::kNHWC; break;
      case mshadow::kCWN: effective_layout = mshadow::kCHWN; break;
      default: break;
    }

    // Double check to make sure this class supports the operation
    if (!Supports(param, forward_compute_type, backward_compute_type, rctx.ctx.dev_id))
      LOG(FATAL) << "Need CuDNN >= 6.0 for dilated deconvolution.";

    InitDescriptors(in_shape, out_shape,
                    cudnn_forward_compute_type, cudnn_backward_compute_type);
    
    GetTempSize(rctx);
    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }
    // In cuDNN_v6, dilated convolution descriptors are compatible with only a
    // single convolution algorithm.  Despite this, we go through the algorithm
    // selection process, which will return the only algorithm supported.  This
    // approach keeps the treatment of convolution cases uniform and will
    // naturally respond to more algorithms supporting dilated convolutions in
    // future cuDNN releases.
  }

  ~CuDNNDeconvolutionOp() {
    MIOPEN_CALL(miopenDestroyTensorDescriptor(in_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(out_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(bias_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(filter_desc_));
    MIOPEN_CALL(miopenDestroyConvolutionDescriptor(forward_conv_desc_));
    MIOPEN_CALL(miopenDestroyConvolutionDescriptor(back_conv_desc_));
    MIOPEN_CALL(miopenDestroyConvolutionDescriptor(back_conv_desc_w_));
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, forward_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    DType *data_ptr = GetNdPtr(in_data[deconv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[deconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *out_ptr = GetNdPtr(out_data[deconv::kOut], param_.kernel.ndim() + 2, s);

    miopenConvAlgoPerf_t bwd_alg_pref;
    bwd_alg_pref.bwd_data_algo = miopenConvolutionBwdDataAlgoGEMM;

    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta  = 0.0f;
       int req_alg_count = 1;
       int retn_algo_count = 0;
    MIOPEN_CALL(miopenFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
               in_desc_,
               data_ptr + data_offset_ * g,
               filter_desc_,
               wmat_ptr + weight_offset_ * g ,
               forward_conv_desc_,
               out_desc_,
               out_ptr + out_offset_ * g,
               req_alg_count,
	       &retn_algo_count,
               &bwd_alg_pref,
               workspace.dptr_,
               workspace_size,
               false));
    back_algo_.Set(bwd_alg_pref.bwd_data_algo, false);

      MIOPEN_CALL(miopenConvolutionBackwardData(s->dnn_handle_,
                 &alpha,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 filter_desc_,
                 wmat_ptr + weight_offset_ * g,
                 forward_conv_desc_,
                 back_algo_.AlgoNumber(), 
                 &beta,
                 out_desc_,
                 out_ptr + out_offset_ * g,
                 workspace.dptr_,
                 workspace_size));

      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[deconv::kBias].get<gpu, 1, DType>(s);
	MIOPEN_CALL(miopenConvolutionForwardBias(s->dnn_handle_,
				                &alpha,
						bias_desc_,
                                                bias.dptr_ + bias_offset_ * g,
                                                &beta,
                                                out_desc_,
                                                out_ptr + out_offset_ * g));
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), param_.no_bias ? 2U : 3U);
    CHECK_EQ(in_grad.size(), expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    // I/O's should have 2 more dims than the kernel dim
    DType *grad_ptr = GetNdPtr(out_grad[deconv::kOut], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[deconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *gwmat_ptr = GetNdPtr(in_grad[deconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *data_ptr = GetNdPtr(in_data[deconv::kData], param_.kernel.ndim() + 2, s);
    DType *gdata_ptr = GetNdPtr(in_grad[deconv::kData], param_.kernel.ndim() + 2, s);

    CHECK_NE(req[deconv::kWeight], kWriteInplace);
    if (!param_.no_bias) {
      CHECK_NE(req[deconv::kBias], kWriteInplace);
    }
    CHECK_NE(req[deconv::kData], kWriteInplace);
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, backward_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType bias_beta = 0.0f;
      typename DataType<DType>::ScaleType data_beta = 0.0f;
      typename DataType<DType>::ScaleType weight_beta = 0.0f;
      if (!param_.no_bias && (req[deconv::kBias] != kNullOp)) {
        Tensor<gpu, 1, DType> gbias = in_grad[deconv::kBias].get<gpu, 1, DType>(s);
       MIOPEN_CALL(miopenConvolutionBackwardBias(s->dnn_handle_,
                                                &alpha,
                                                out_desc_,
                                                grad_ptr + out_offset_ * g,
                                                &bias_beta,
                                                bias_desc_,
                                                gbias.dptr_ + bias_offset_ * g));
      }
      if (req[deconv::kWeight] != kNullOp) {

        miopenConvAlgoPerf_t bwd_alg_pref;

    bwd_alg_pref.bwd_weights_algo = miopenConvolutionBwdWeightsAlgoGEMM;

    int req_alg_count = 1;
    int retn_algo_count = 0;
    MIOPEN_CALL(miopenFindConvolutionBackwardWeightsAlgorithm(s->dnn_handle_,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 out_desc_,
                 grad_ptr + out_offset_  * g,
                 back_conv_desc_,
                 filter_desc_,
                 gwmat_ptr + weight_offset_ * g,
                 req_alg_count,
		 &retn_algo_count,
                 &bwd_alg_pref,
                 workspace.dptr_,
                 workspace_size,
                 false));
     back_algo_w_.Set(bwd_alg_pref.bwd_weights_algo, false);
        MIOPEN_CALL(miopenConvolutionBackwardWeights(
          s->dnn_handle_,
          &alpha,
          in_desc_,
          data_ptr + data_offset_ * g,
          out_desc_,
          grad_ptr + out_offset_ * g,
          back_conv_desc_,
          back_algo_w_.AlgoNumber(),
          &weight_beta,
          filter_desc_,
          gwmat_ptr + weight_offset_ * g,
          workspace.dptr_,
          workspace_size));
      }
      if (req[deconv::kData] != kNullOp) {


      miopenConvAlgoPerf_t fwd_algo_pref;

           fwd_algo_pref.fwd_algo = miopenConvolutionFwdAlgoGEMM;
       int req_algo_count = 1 ;
       int retn_algo_count =0 ;
           MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(s->dnn_handle_,
                  out_desc_,
                  grad_ptr + out_offset_ * g,
                  filter_desc_,
                  wmat_ptr + weight_offset_ * g,
                  back_conv_desc_,
                  in_desc_,
                  gdata_ptr + data_offset_ * g,
                  req_algo_count,
		  &retn_algo_count,
                  &fwd_algo_pref,
                  (void*)workspace.dptr_,
                  workspace_size,
                  false));
             forward_algo_.Set(fwd_algo_pref.fwd_algo, false);

         MIOPEN_CALL(miopenConvolutionForward(s->dnn_handle_,
                                           &alpha,
                                           out_desc_,
                                           grad_ptr + out_offset_ * g,
                                           filter_desc_,
                                           wmat_ptr + weight_offset_ * g,
                                           back_conv_desc_,
                                           forward_algo_.AlgoNumber(), 
                                           &data_beta,
                                           in_desc_,
                                           gdata_ptr + data_offset_ * g,
                                           workspace.dptr_,
                                           workspace_size));
      }
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the deconvolution
 * operation described by `param`: cuDNN v5 and earlier does not support
 * dilated convolutions.
 */
  static bool Supports(DeconvolutionParam param,
                       int forward_compute_type,
                       int backward_compute_type,
                       int dev_id) {
    using namespace mshadow;

    // NDHWC not supported, NHWC not supported in true fp16
    auto layout_val = param.layout.value();
    auto true_fp16 = DataType<DType>::kFlag == kFloat16 &&
      (forward_compute_type == kFloat16 || backward_compute_type == kFloat16);
    if (layout_val == kNDHWC || layout_val == kNWC ||
        layout_val == kNHWC && true_fp16)
      return false;

    // Permits graceful fallback to pseudo-fp16 on heterogenous systems
    if (!SupportsFloat16Compute(dev_id) &&
        (forward_compute_type == kFloat16 || backward_compute_type == kFloat16)) {
      return false;
    }

    // The factor by which the effective filter size grows based on dilation.
    auto filterDilationFactor = param.dilate.Size();

    // The v6 kernels that backprop a dilated convolution don't handle fp16.
    // Since the deconvolution "forward" kernel is really a backprop-to-data
    // cuDNN kernel, the following logic is slightly different than that
    // used in CuDNNConvolution::Supports().

    // Dilation support across all architectures only available after v6.0.20.
    return filterDilationFactor == 1 ||
           filterDilationFactor > 1  &&
           (backward_compute_type != kFloat16) &&
           (forward_compute_type != kFloat16);
  }

 private:
/*!
 * \brief Translate an mxnet datatype to the corresponding cudnnDataType_t.
 */
  miopenDataType_t convertToCuDNNDataType(int dtype) {
    miopenDataType_t converted = miopenFloat;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_REAL_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

  inline void InitDescriptors(const std::vector<TShape> &in_shape,
                              const std::vector<TShape> &out_shape,
                              miopenDataType_t cudnn_forward_compute_type,
                              miopenDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);

    TShape dshape = in_shape[deconv::kData];
    TShape wshape = in_shape[deconv::kWeight];
    TShape oshape = out_shape[deconv::kOut];
    TShape dstride, ostride, wstride;
    wshape[0] /= param_.num_group;
    if (param_.kernel.ndim() == 1 || param_.kernel.ndim() == 2) {
      // 1d or 2d conv
      index_t o_pad[2];
      index_t o_adj[2];
      if (param_.kernel.ndim() == 2) {
        param_.InferPad(dshape, o_pad, o_adj);
      } else {
        index_t o_pad_1D[1];
        index_t o_adj_1D[1];
        param_.InferPad(dshape, o_pad_1D, o_adj_1D);
        o_pad[0] = 0;
        o_pad[1] = o_pad_1D[0];
      }
      auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
      auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});

      MIOPEN_CALL(miopenInitConvolutionDescriptor(forward_conv_desc_,
                                                 miopenConvolution,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilate[0],
                                                 dilate[1]));
      MIOPEN_CALL(miopenInitConvolutionDescriptor(back_conv_desc_,
                                                 miopenConvolution,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilate[0],
                                                 dilate[1]));
      MIOPEN_CALL(miopenInitConvolutionDescriptor(back_conv_desc_w_,
                                                 miopenConvolution,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilate[0],
                                                 dilate[1]));
      if (param_.kernel.ndim() == 2) {
        wstride = ConvertLayout(Strides<4>(wshape), param_.layout.value(), kNCHW);
        wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
        dstride = ConvertLayout(Strides<4>(dshape), param_.layout.value(), kNCHW);
        dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        ostride = ConvertLayout(Strides<4>(oshape), param_.layout.value(), kNCHW);
        oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
      } else {
        wstride = ConvertLayout(Strides<3>(wshape), param_.layout.value(), kNCW);
        wstride = TShape({wstride[0], wstride[1], wstride[1], wstride[2]});
        wshape = ConvertLayout(wshape.get<3>(), param_.layout.value(), kNCW);
        wshape = TShape({wshape[0], wshape[1], 1, wshape[2]});
        dstride = ConvertLayout(Strides<3>(dshape), param_.layout.value(), kNCW);
        dstride = TShape({dstride[0], dstride[1], dstride[1], dstride[2]});
        dshape = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
        dshape = TShape({dshape[0], dshape[1], 1, dshape[2]});
        ostride = ConvertLayout(Strides<3>(oshape), param_.layout.value(), kNCW);
        ostride = TShape({ostride[0], ostride[1], ostride[1], ostride[2]});
        oshape = ConvertLayout(oshape.get<3>(), param_.layout.value(), kNCW);
        oshape = TShape({oshape[0], oshape[1], 1, oshape[2]});
      }
      MIOPEN_CALL(miopenSet4dTensorDescriptor(filter_desc_,
                                            dtype_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      index_t o_pad[3];
      index_t o_adj[3];
      param_.InferPad(dshape, o_pad, o_adj);

      CHECK_EQ(param_.layout.value(), kNCDHW) << "CuDNN only support 3D conv with NCDHW layout";
      std::vector<int> wshape_buffer(wshape.ndim());
       int dimensn = static_cast<int>(wshape.ndim());
      if (dimensn > 0 && dimensn < 6)
{
      wstride = ConvertLayout(Strides<5>(wshape), param_.layout.value(), kNCDHW);
      MIOPEN_CALL(miopenSetTensorDescriptor(filter_desc_,
                                        dtype_,
                                        static_cast<int>(wshape.ndim()),
                                        CastTShapeToIntPtr(wshape, &wshape_buffer),
                                        reinterpret_cast<int*>(&wstride[0]))); 
}

      MIOPEN_CALL(miopenInitConvolutionNdDescriptor(forward_conv_desc_,
                                                  param_.kernel.ndim(),
                                                  reinterpret_cast<int*>(&o_pad[0]),
						  param_stride_.data(),
                                                  param_dilate_.data(),
                                                  miopenConvolution));
       MIOPEN_CALL(miopenInitConvolutionNdDescriptor(back_conv_desc_,
                                                  param_.kernel.ndim(),
                                                  reinterpret_cast<int*>(&o_pad[0]),
						  param_stride_.data(),
                                                  param_dilate_.data(),
                                                  miopenConvolution));
       MIOPEN_CALL(miopenInitConvolutionNdDescriptor(back_conv_desc_w_,
                                                  param_.kernel.ndim(),
                                                  reinterpret_cast<int*>(&o_pad[0]),
						  param_stride_.data(),
                                                  param_dilate_.data(),
                                                  miopenConvolution));


      dstride = ConvertLayout(Strides<5>(dshape), param_.layout.value(), kNCDHW);
      dshape = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
      ostride = ConvertLayout(Strides<5>(oshape), param_.layout.value(), kNCDHW);
      oshape = ConvertLayout(oshape.get<5>(), param_.layout.value(), kNCDHW);
    
   }
    dshape[1] /= param_.num_group;
    oshape[1] /= param_.num_group;
    weight_offset_ = wshape.Size();
    data_offset_ = dstride[1] * dshape[1];
    out_offset_ = ostride[1] * oshape[1];

    std::vector<int> dshape_buffer(dshape.ndim());
    std::vector<int> dstride_buffer(dstride.ndim());
    int dimensn = static_cast<int>(dshape.ndim());
    if (dimensn > 0 && dimensn < 6)
      MIOPEN_CALL(miopenSetTensorDescriptor(in_desc_,
                                        dtype_,
                                        static_cast<int>(dshape.ndim()),
                                        CastTShapeToIntPtr(dshape, &dshape_buffer),
                                        CastTShapeToIntPtr(dstride, &dstride_buffer)));

    std::vector<int> oshape_buffer(oshape.ndim());
    std::vector<int> ostride_buffer(ostride.ndim());

     int ndim = static_cast<int>(oshape.ndim());
    if (ndim > 0 && ndim < 6)
      MIOPEN_CALL(miopenSetTensorDescriptor(out_desc_,
                                        dtype_,
                                        static_cast<int>(oshape.ndim()),
                                        CastTShapeToIntPtr(oshape, &oshape_buffer),
                                        CastTShapeToIntPtr(ostride, &ostride_buffer))); 

    if (!param_.no_bias) {
      TShape bias = in_shape[deconv::kBias];
      bias_offset_ = bias[0] / param_.num_group;
      std::vector<int> bias_shape = {1,
                                     static_cast<int>(bias[0] / param_.num_group),
                                     1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias_offset_), 1, 1, 1};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(1);
      }
    int dims = static_cast<int>(bias_shape.size());
    if (dims > 0 && dims < 6)
      MIOPEN_CALL(miopenSetTensorDescriptor(bias_desc_,
                                        dtype_,
                                        static_cast<int>(bias_shape.size()),
                                        &bias_shape[0],
                                        &bias_stride[0]));    
   }
 }



  void GetTempSize(const RunContext& ctx) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t back_data_algo_workspace_size = 0;
    size_t back_filter_algo_workspace_size = 0;
    size_t forward_algo_workspace_size = 0;
    MIOPEN_CALL(miopenConvolutionBackwardDataGetWorkSpaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               forward_conv_desc_,
               out_desc_,
               &back_data_algo_workspace_size));
    MIOPEN_CALL(miopenConvolutionBackwardWeightsGetWorkSpaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               back_conv_desc_,
               filter_desc_,
               &back_filter_algo_workspace_size));
     MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               back_conv_desc_,
               in_desc_,
               &forward_workspace_byte_));


    forward_workspace_byte_ = back_data_algo_workspace_size;
    backward_workspace_byte_ = std::max(forward_algo_workspace_size,
                                        back_filter_algo_workspace_size);
  }

  int *CastTShapeToIntPtr(const TShape& s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = NULL;
    if (dim == 3) {
      Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

  // Converts a TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const TShape &s) {
    uint32_t ndim = s.ndim();
    TShape strides(ndim);
    for (uint32_t i = 0; i != ndim; ++i)
      strides[i] = s.ProdShape(i+1, ndim);
    return strides.get<dim>();
  }

  void InitBufferForParam() {
    CastTShapeToIntPtr(param_.stride, &param_stride_);
    CastTShapeToIntPtr(param_.dilate, &param_dilate_);
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx, size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words = size_bytes / sizeof(DType) + 1;
    return ctx.requested[deconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(size_words), s);
  }

  // Returns the size in bytes of the 1D Tensor of words.
  size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, DType> &tensor) {
    return tensor.MSize() * sizeof(DType);
  }


  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const TShape &dshape) {
    int c = 0;
    switch (dshape.ndim()) {
      case 3: c = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW)[1]; break;
      case 4: c = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW)[1]; break;
      case 5: c = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW)[1]; break;
      default:
        LOG(FATAL) << "Unexpected deconvolution data dimension " << dshape.ndim();
    }
    return c;
  }

  // Make a number of allocations and directly free them, ensuring room for an equivalent set of
  // hipMalloc() calls by (say) cudnnFind().  `elements` spec the alloc size in DTypes, not bytes.
  void ReserveElements(const std::vector<size_t> &elements) {
    std::vector<Storage::Handle> handles;
    for (size_t alloc_element : elements)
        handles.push_back(Storage::Get()->Alloc(alloc_element * sizeof(DType), Context::GPU()));
    for (auto &handle : handles)
        Storage::Get()->DirectFree(handle);
  }


  // Log that no suitable algo was found that met the workspace constraints, then exit.
  void LogNoSuitableAlgoAndExit(int num_algos_tried, size_t min_memory_needs,
                                size_t workspace_byte, std::string algo_kind) {
    LOG(FATAL) << num_algos_tried << " " << algo_kind << " with minimum memory requirement "
               << min_memory_needs << " bytes have been tried. Workspace size is set to "
               << workspace_byte << " bytes, please consider reducing the batch/model size, "
               << "or increasing workspace size.";
  }

  std::vector<int> param_stride_;
  std::vector<int> param_dilate_;

  int forward_compute_type_;
  int backward_compute_type_;
  const std::vector<TShape> in_shapes_;
  const std::vector<TShape> out_shapes_;

  // Temp workspace size in bytes needed for Forward() operation.  Note that
  // in deconvolution, this is handled by the cuDNN backprop-to-data kernel.
  size_t forward_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() operation.  Note that
  // in deconvolution, this is handled by the cuDNN forward kernel and the
  // the cuDNN backprop-to-filter kernel.
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  miopenDataType_t dtype_;
  miopenTensorDescriptor_t in_desc_;
  miopenTensorDescriptor_t out_desc_;
  miopenTensorDescriptor_t bias_desc_;
  miopenTensorDescriptor_t filter_desc_;
  // Convolution descriptor for "forward" inference operation.
  // Note that in deconvolution, the forward operation is handled
  // by the cuDNN backprop-to-data kernel.
  miopenConvolutionDescriptor_t forward_conv_desc_;
  // Convolution descriptor for "back-prop" operations to data .
  // Note that in deconvolution, the backprop-to-data operation is handled
  // by the cuDNN forward kernel.
  miopenConvolutionDescriptor_t back_conv_desc_;
  // Convolution descriptor for "back-prop" operations to filter.
  // Note that in deconvolution, the backprop-to-data operation is handled
  // by the backprop-to-filter kernel (so consistent with the treatment
  // in convolution).
  miopenConvolutionDescriptor_t back_conv_desc_w_;
  // Algorithm for the cuDNN forward kernel (used in gradient backprop to input)
  CuDNNAlgo<miopenConvFwdAlgorithm_t> forward_algo_;
  // Algorithm for the cuDNN backprop-to-data kernel (used in inference)
  CuDNNAlgo<miopenConvBwdDataAlgorithm_t> back_algo_;
  // Algorithm for the cuDNN backprop-to-filter kernel
  CuDNNAlgo<miopenConvBwdWeightsAlgorithm_t> back_algo_w_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;
  // Is req[kWeight] == deconv::kAddTo ?
  bool add_to_weight_;
  DeconvolutionParam param_;
};
#endif  // MXNET_USE_MIOPEN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_
