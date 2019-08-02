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
#if defined(__HIP_PLATFORM_NVCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
}
hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle,
                                   int n,
                                   float *Aarray[],
                                   int lda,
                                   int * PivotArray,
                                   int *infoArray,
                                   int batchSize)
{
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasSgetrfBatched((cublasHandle_t) handle,
                                    n,
                                   *Aarray[],
                                   lda,
                                   *PivotArray,
                                   *infoArray,
                                    batchSize));*/
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
}

hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle,
                                   int n,
                                   double *Aarray[],
                                   int lda,
                                   int *PivotArray,
                                   int *infoArray,
                                   int batchSize)
{
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasDgetrfBatched((cublasHandle_t) handle,
                                   n,
                                   *Aarray[],
                                   lda,
                                   *PivotArray,
                                   *infoArray,
                                   batchSize);*/
return HIPBLAS_STATUS_NOT_SUPPORTED;

#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
}
hipblasStatus_t hipblasSgetriBatched(hipblasHandle_t handle,
                                   int n,
                                   const float *Aarray[],
                                   int lda,
                                   const int *PivotArray,
                                   float *Carray[],
                                   int ldc,
                                   int *infoArray,
                                   int batchSize)
{
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasSgetriBatched((cublasHandle_t) handle,
                                    n,
                                   *Aarray[],
                                   lda,
                                   *PivotArray,
                                   *Carray[],
                                   ldc,
                                   *infoArray,
                                   batchSize); */
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
}

hipblasStatus_t hipblasDgetriBatched(hipblasHandle_t handle,
                                   int n,
                                   const double *Aarray[],
                                   int lda,
                                   const int *PivotArray,
                                   double *Carray[],
                                   int ldc,
                                   int *infoArray,
                                   int batchSize)
{
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasDgetriBatched((cublasHandle_t) handle,
                                   n,
                                   *Aarray[],
                                   lda,
                                   *PivotArray,
                                   *Carray[],
                                   ldc,
                                   *infoArray,
                                   batchSize);*/
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
}

hipblasStatus_t hipblasGemmStridedBatchedEx (hipblasHandle_t handle,
                                             hipblasOperation_t transa,
                                             hipblasOperation_t transb,
                                             int m,
                                             int n,
                                             int k,
                                          const void    *alpha,
                                          const void     *A,
                           		  hipblasDatatype_t Atype,
                           		  int lda,
                           		  long long int strideA,
                           		  const void     *B,
                            		  hipblasDatatype_t Btype,
                             		  int ldb,
                           		  long long int strideB,
                           		  const void    *beta,
                               		  void           *C,
                          		  hipblasDatatype_t Ctype,
                         		  int ldc,
                         		  long long int strideC,
                         		  int batchCount,
                           		  hipblasDatatype_t computeType,
                           	          hipblasGemmAlgo_t algo)
{
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasGemmStridedBatchedEx((cublasHandle_t)handle,
                           hipOperationToCudaOperation(transa), 
                           hipOperationToCudaOperation(transb),
                           m, 
                           n, 
                           k,
                           *alpha,
                           *A, 
                           Atype, 
                           lda,
                           strideA,
                           *B, 
                           Btype, 
                           ldb,
                           strideB,
                           *beta,
                           *C, 
                           Ctype, 
                           ldc,
                           strideC,
                           batchCount,
                           computeType, 
                           algo); */
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
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
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasStrmm((cublasHandle_t) handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           hipOperationToCudaOperation(trans), cublasDiagType_t diag,
                            m,  n,
                           *alpha,
                           *A,  lda,
                           *B,  ldb,
                          *C,  ldc));*/
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)  
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
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
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasDtrmm((cublasHandle_t) handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           hipOperationToCudaOperation(trans), cublasDiagType_t diag,
                            m,  n,
                            *alpha,
                            *A, lda,
                            *B, ldb,
                            *C, ldc));*/
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
  }
 hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle,
                           hipblasFillMode_t uplo, 
			   hipblasOperation_t trans,
                           int n, 
			   int k,
                           const float  *alpha,
                           const float  *A, 
			   int lda,
                           const float  *beta,
                           float        *C, 
			   int ldc)
{
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasSsyrk((cublasHandle_t) handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           n,  k,
                           *alpha,
                           *A, lda,
                           *beta,
                           *C, ldc));*/
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
#if defined(__HIP_PLATFORM_HCC__)
return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
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
#if defined(__HIP_PLATFORM_NVCC__)
/*return hipCUBLASStatusToHIPStatus(cublasDsyrk((cublasHandle_t) handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                            n,  k,
                            *alpha,
                            *A, lda,
                            *beta,
                            *C, ldc));*/
return HIPBLAS_STATUS_NOT_SUPPORTED;
  #endif
  #if defined(__HIP_PLATFORM_HCC__)
  return HIPBLAS_STATUS_NOT_SUPPORTED;
  #endif
  }
