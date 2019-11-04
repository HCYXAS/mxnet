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
 *  Copyright (c) 2018 by Contributors
 * \file adamw.cu
 * \brief Optimizer operators
 * \author Haibin Lin, Moises Hernandez, Andrei Ivanov
 */
#include "./adamw-inl.h"

namespace mxnet {
namespace op {

template<>
void GetScaleFloat<gpu>(const TBlob &scale_blob, float *pScalef) {
  MSHADOW_REAL_TYPE_SWITCH(scale_blob.type_flag_, DType, {
    DType scale = 0;
    CUDA_CALL(hipMemcpy(&scale, scale_blob.dptr<DType>(), sizeof(DType),
                         hipMemcpyDeviceToHost));
    *pScalef = static_cast<float>(scale);
  })
}

NNVM_REGISTER_OP(_adamw_update)
.set_attr<FCompute>("FCompute<gpu>", MPUpdate<gpu, AdamWUpdate<gpu>>);

NNVM_REGISTER_OP(_mp_adamw_update)
.set_attr<FCompute>("FCompute<gpu>", MPUpdate<gpu, MPAdamWUpdate<gpu>>);

NNVM_REGISTER_OP(_multi_adamw_update)
.set_attr<FCompute>("FCompute<gpu>", multiMPUpdate<gpu, false>);

NNVM_REGISTER_OP(_multi_mp_adamw_update)
.set_attr<FCompute>("FCompute<gpu>", multiMPUpdate<gpu, true>);

}  // namespace op
}  // namespace mxnet
