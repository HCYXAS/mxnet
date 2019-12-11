/* Dummy header file to add types and fucntions to silence compiler errors.
   This needs to be replaced with working fixes in actual header files and         libraries.
*/

#ifndef HIPWRAPPERS_H
#define HIPWRAPPERS_H

#if defined(__HIPCC__)
#include <hipblas.h>
#include <hiprand.h>
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"


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
                                            int batchCount);

hipblasStatus_t hipblasSgemmEx  (hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float *alpha, // host or device pointer
                                 const void *A,
                                 hipblasDatatype_t Atype,
                                 int lda,
                                 const void *B,
                                 hipblasDatatype_t Btype,
                                 int ldb,
                                 const float *beta,
                                 void *C,
                                 hipblasDatatype_t Ctype,
                                 int ldc);

hipblasStatus_t hipblasStrmm (hipblasHandle_t handle,
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
                              int ldc);

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
			     int ldc);

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
			    int ldb);

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
			     int ldb);

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
			     int ldc);

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
			     int ldc);
hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t handle,
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
                           hipblasGemmAlgo_t algo); 
#endif
#endif //HIPWRAPPERS_H
