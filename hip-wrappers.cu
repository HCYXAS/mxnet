/* Dummy header file to add types and fucntions to silence compiler errors.
   This needs to be replaced with working fixes in actual header files and libraries.
*/

#include "hip/hip_runtime.h"
#include "hip-wrappers.h"



hipblasStatus_t  hipblasHgemmStridedBatched(hipblasHandle_t handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int m,
					    int n,
					    int k,
                                            const __half           *alpha,
                                            const __half           *A,
					    int lda,
                                            long long int          strideA,
                                            const __half           *B,
					    int ldb,
                                            long long int          strideB,
                                            const __half           *beta,
                                            __half                 *C,
					    int ldc,
                                            long long int          strideC,
                                            int batchCount)
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
				  hipblasDatatype_t Atype,
                                   int lda,
                                   const void *B,
                                   hipblasDatatype_t Btype,
                                  int ldb,
                                  const float *beta,
				  void *C,
                                   hipblasDatatype_t Ctype,
                                   int ldc)
  {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

 hipblasStatus_t hipblasStrmm (void *h,//hipblasHandle_t handle,
                              hipblasSideMode_t rightside,
                              hipblasFillMode_t lower,
                              hipblasOperation_t transpose,
                              hipblasDiagType_t diag,
                              int m,
                              int n,
                              const float *alpha,
                              const float *A,
                              int lda,
                              const float *B,
                              int ldb,
                              float *C,
                              int ldc)
  {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

 hipblasStatus_t hipblasDtrmm(hipblasHandle_t handle,
                             hipblasSideMode_t side,
                             hipblasFillMode_t uplo,
                             hipblasOperation_t trans,
                             hipblasDiagType_t diag,
                             int m,
                             int n,
                             const double *alpha,
                             const double *A,
                             int lda,
                             const double *B,
                             int ldb,
                             double *C,
                             int ldc)
  {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

hipblasStatus_t hipblasStrsm(hipblasHandle_t handle,
                            hipblasSideMode_t side, 
                            hipblasFillMode_t uplo,
                            hipblasOperation_t trans, 
                            hipblasDiagType_t diag,
                            int m, 
                            int n,
                            const float *alpha,
                            const float *A, 
                            int lda,
                            float *B, 
                            int ldb)
  {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

 hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle,
                             hipblasSideMode_t side, 
                             hipblasFillMode_t uplo,
                             hipblasOperation_t trans, 
                             hipblasDiagType_t diag,
                             int m, 
                             int n,
                             const double *alpha,
                             const double *A,
                             int lda,
                             double *B, 
                             int ldb)
  {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

 hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle,
                             hipblasFillMode_t uplo, 
                             hipblasOperation_t trans,
                             int n, 
                             int k,
                             const float *alpha,
                             const float *A, 
                             int lda,
                             const float *beta,
                             float *C, 
                             int ldc)
  {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
 
 hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle,
                             hipblasFillMode_t uplo, 
                             hipblasOperation_t trans,
                             int n, 
                             int k,
                             const double *alpha,
                             const double *A, 
                             int lda,
                             const double *beta,
                             double *C, 
                             int ldc)
  {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
