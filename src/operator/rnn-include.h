
#include "./operator_common.h"
#include "./rnn_impl.h"
#include "./mxnet_op.h"
#if MSHADOW_USE_MIOPEN == 1 || MSHADOW_USE_GPU == 1
#include "./miopen_rnn-inl.h"
#endif
#if MSHADOW_USE_CUDNN == 1 || MSHADOW_USE_GPU == 1
#include "./rnn-inl.h"
#endif

