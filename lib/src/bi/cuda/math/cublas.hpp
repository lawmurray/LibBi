/**
 * @file
 *
 * Macros for CUBLAS usage.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_CUBLAS_HPP
#define BI_CUDA_CUBLAS_HPP

#include "../../misc/assert.hpp"

#include "cublas.h"

#include "boost/typeof/typeof.hpp"

/**
 * @def CUBLAS_CHECKED_CALL(call)
 *
 * Make CUBLAS function call and assert success.
 *
 * @arg Function call.
 */
#if defined(__CUDACC__) && !defined(NDEBUG)
#define CUBLAS_CHECKED_CALL(call) \
  { \
    cublasStatus cublasErr; \
    cublasErr = call; \
    BI_ASSERT(cublasErr == CUBLAS_STATUS_SUCCESS, "CUBLAS Error " << cublasErr); \
  }
#else
#define CUBLAS_CHECKED_CALL(call) call
#endif

/**
 * @def CUBLAS_CHECK
 *
 * Check for CUBLAS errors and fail if detected.
 */
#if defined(__CUDACC__) && !defined(NDEBUG)
#define CUBLAS_CHECK \
  { \
    cublasStatus cublasErr; \
    cublasErr = cublasGetError(); \
    BI_ASSERT(cublasErr == CUBLAS_STATUS_SUCCESS, "CUBLAS Error " << cublasErr); \
  }
#else
#define CUBLAS_CHECK
#endif

/**
 * @def CUBLAS_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for cublas functions.
 */
#define CUBLAS_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct cublas_##name {}; \
  \
  template<> \
  struct cublas_##name<float> { \
    static BOOST_TYPEOF(cublas##sname) *func; \
  }; \
  \
  template<> \
  struct cublas_##name<double> { \
    static BOOST_TYPEOF(cublas##dname) *func; \
  }; \
}

CUBLAS_FUNC(copy, Dcopy, Scopy)
CUBLAS_FUNC(scal, Dscal, Sscal)
CUBLAS_FUNC(dot, Ddot, Sdot)
CUBLAS_FUNC(iamax, Idamax, Isamax)
CUBLAS_FUNC(axpy, Daxpy, Saxpy)
CUBLAS_FUNC(symv, Dsymv, Ssymv)
CUBLAS_FUNC(gemv, Dgemv, Sgemv)
CUBLAS_FUNC(trmv, Dtrmv, Strmv)
CUBLAS_FUNC(gbmv, Dgbmv, Sgbmv)
CUBLAS_FUNC(gemm, Dgemm, Sgemm)
CUBLAS_FUNC(symm, Dsymm, Ssymm)
CUBLAS_FUNC(trmm, Dtrmm, Strmm)
CUBLAS_FUNC(ger, Dger, Sger)
CUBLAS_FUNC(syr, Dsyr, Ssyr)
CUBLAS_FUNC(syr2, Dsyr2, Ssyr2)
CUBLAS_FUNC(syrk, Dsyrk, Ssyrk)
CUBLAS_FUNC(trsv, Dtrsv, Strsv)
CUBLAS_FUNC(trsm, Dtrsm, Strsm)

#endif
