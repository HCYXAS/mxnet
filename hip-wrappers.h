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
                                int ldc);

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
#endif //HIPWRAPPERS_H
