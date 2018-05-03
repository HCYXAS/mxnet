#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#if defined(USE_CUDA) && !defined (USE_HIP)
#include "cuda_backend/gpu_runtime.h"
#elif defined(USE_HIP) && !defined (USE_CUDA)

#else
	#error("Must define exactly one of USE_CUDA or USE_HIP");
#endif


#endif
