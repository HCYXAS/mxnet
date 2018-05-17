/* Dummy header file to add types and fucntions to silence compiler errors.
   This needs to be replaced with working fixes in actual header files and         libraries.
*/

#ifndef HIPWRAPPERS_H
#define HIPWRAPPERS_H

#include <hipblas.h>
#include <hiprand.h>
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"


#if defined(__HIP_PLATFORM_HCC__) && !defined (__HCC__)
typedef struct {
   unsigned short x;
}__half;
#endif

typedef enum hipDataType_t
{
    HIP_R_16F= 2,  /* real as a half */
    HIP_C_16F= 6,  /* complex as a pair of half numbers */
    HIP_R_32F= 0,  /* real as a float */
    HIP_C_32F= 4,  /* complex as a pair of float numbers */
    HIP_R_64F= 1,  /* real as a double */
    HIP_C_64F= 5,  /* complex as a pair of double numbers */
    HIP_R_8I = 3,  /* real as a signed char */
    HIP_C_8I = 7,  /* complex as a pair of signed char numbers */
    HIP_R_8U = 8,  /* real as a unsigned char */
    HIP_C_8U = 9,  /* complex as a pair of unsigned char numbers */
    HIP_R_32I= 10, /* real as a signed int */
    HIP_C_32I= 11, /* complex as a pair of signed int numbers */
    HIP_R_32U= 12, /* real as a unsigned int */
    HIP_C_32U= 13  /* complex as a pair of unsigned int numbers */
} hipDataType;

hipblasStatus_t hipblasSgemmEx  (hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float *alpha, // host or device pointer
                                 const void *A,
                                 hipDataType Atype,
                                 int lda,
                                 const void *B,
                                 hipDataType Btype,
                                 int ldb,
                                 const float *beta,
                                 void *C,
                                 hipDataType Ctype,
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
#endif //HIPWRAPPERS_H
