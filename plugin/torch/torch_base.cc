/*!
 * Copyright (c) 2016 by Contributors
 * \file torch_base.cc
 * \brief torch_state
 * \author Junyuan Xie
*/
#include "./torch_base.h"

namespace mxnet {
TorchState::TorchState() {
  this->L = luaL_newstate();

  luaL_openlibs(L);
  luaL_loadstring(L,
                  "require 'torch'\n"
                  "require 'nn'\n"
#if MXNET_USE_GPU
                  "require 'cutorch'\n"
                  "require 'cunn'\n"
#if MXNET_USE_CUDNN
                  "require 'cudnn'\n"
#elif MXNET_USE_MIOPEN
		  "require 'miopen'\n"
#endif  // MXNET_USE_CUDNN
#endif  // MXNET_USE_GPU
                  ); // NOLINT(*)
  int err = lua_pcall(L, 0, 0, 0);
  CHECK_EQ(err, 0) << lua_tostring(L, -1);
}

TorchState* TorchState::ThreadSharedLuaState() {
  thread_local TorchState* state = nullptr;
  if (!state) {
    state = new TorchState();
  }
  return state;
}

template<>
void TorchState::SetStream(mshadow::Stream<mshadow::cpu>* s) {
  return;
}

#if MXNET_USE_GPU
template<>
void TorchState::SetStream(mshadow::Stream<mshadow::gpu>* s) {
  CudaState()->currentStream = mshadow::Stream<gpu>::GetStream(s);
}
#endif  // MXNET_USE_GPU
}  // namespace mxnet
