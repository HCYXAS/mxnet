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
 *  Copyright (c) 2015 by Contributors
 * \file dist_common.cuh
 * \brief Function definition of common functions for distributions
 * \with two parameters.
 */

#include "./dist_common.h"

namespace mxnet {
namespace op {

template <>
void _copy<gpu>(float *dst, float *src) {
CUDA_CALL(hipMemcpy(dst, src, sizeof(float), hipMemcpyDeviceToHost));
}

template <>
void _copy<gpu>(double *dst, double *src) {
CUDA_CALL(hipMemcpy(dst, src, sizeof(double), hipMemcpyDeviceToHost));
}

}  // namespace op
}  // namespace mxnet
