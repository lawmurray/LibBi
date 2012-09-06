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

#include "../cuda.hpp"
#include "../../misc/assert.hpp"

#include "cublas_v2.h"

#include "boost/typeof/typeof.hpp"

namespace bi {
/*
 * Mappings from character arguments to CUBLAS enums.
 */
inline cublasFillMode_t cublas_uplo(const char uplo) {
  /* pre-condition */
  BI_ASSERT(uplo == 'U' || uplo == 'L');

  return (uplo == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

inline cublasOperation_t cublas_trans(const char trans) {
  /* pre-condition */
  BI_ASSERT(trans == 'N' || trans == 'T');

  return (trans == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
}

inline cublasSideMode_t cublas_side(const char side) {
  /* pre-condition */
  BI_ASSERT(side == 'L' || side == 'R');

  return (side == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
}

inline cublasDiagType_t cublas_diag(const char diag) {
  /* pre-condition */
  BI_ASSERT(diag == 'N' || diag == 'U');

  return (diag == 'N') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
}

/**
 * Synchronize with device.
 *
 * @param handle CUBLAS handle.
 *
 * Synchronizes on the stream associated with the CUBLAS handle.
 */
inline void synchronize(cublasHandle_t handle) {
  #ifdef ENABLE_CUDA
  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  synchronize(stream);
  #endif
}

}

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
    cublasStatus_t cublasErr; \
    cublasErr = call; \
    BI_ASSERT_MSG(cublasErr == CUBLAS_STATUS_SUCCESS, "CUBLAS Error " << cublasErr); \
  }
#else
#define CUBLAS_CHECKED_CALL(call) call
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
