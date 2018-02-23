/* Dummy header file to add types and fucntions to silence compiler errors.
   This needs to be replaced with working fixes in actual header files and libraries.
*/

#include "hip/hip_runtime.h"
#include "hip-wrappers.h"

 hipblasStatus_t hipblasHgemm    (hipblasHandle_t handle,
                                hipblasOperation_t transa,
                                hipblasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const __half *alpha, // host or device pointer
                                const __half *A,
                                int lda,
                                const __half *B,
                                int ldb,
                                const __half *beta, // host or device pointer
                                __half *C,
                                int ldc)
 {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
 }

 hipblasStatus_t hipblasSgemmEx  (hipblasHandle_t handle,
                                  hipblasOperation_t transa,
                                  hipblasOperation_t transb,
				  int m,
                                  int n,
                                  int k,
				  const float *alpha,
				  const void *A,
				  hipDataType Atype,
                                   int lda,
                                   const void *B,
                                   hipDataType Btype,
                                  int ldb,
                                  const float *beta,
				  void *C,
                                   hipDataType Ctype,
                                   int ldc)
  {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

