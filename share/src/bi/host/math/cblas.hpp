/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_CBLAS_HPP
#define BI_HOST_MATH_CBLAS_HPP

#include "../../misc/assert.hpp"

extern "C" {
#ifdef HAVE_MKL_CBLAS_H
#include "mkl_cblas.h"
#elif HAVE_CBLAS_H
#include "cblas.h"
#elif HAVE_GSL_GSL_CBLAS_H
#include "gsl/gsl_cblas.h"
#endif
}

#include "boost/typeof/typeof.hpp"

namespace bi {
/*
 * Mappings from character arguments to CBLAS enums.
 */
inline CBLAS_UPLO cblas_uplo(const char uplo) {
  /* pre-condition */
  BI_ASSERT(uplo == 'U' || uplo == 'L');

  return (uplo == 'U') ? CblasUpper : CblasLower;
}

inline CBLAS_TRANSPOSE cblas_trans(const char trans) {
  /* pre-condition */
  BI_ASSERT(trans == 'N' || trans == 'T');

  return (trans == 'N') ? CblasNoTrans : CblasTrans;
}

inline CBLAS_SIDE cblas_side(const char side) {
  /* pre-condition */
  BI_ASSERT(side == 'L' || side == 'R');

  return (side == 'L') ? CblasLeft : CblasRight;
}

inline CBLAS_DIAG cblas_diag(const char diag) {
  /* pre-condition */
  BI_ASSERT(diag == 'N' || diag == 'U');

  return (diag == 'N') ? CblasNonUnit : CblasUnit;
}

}

/**
 * @def CBLAS_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for cblas functions.
 */
#define CBLAS_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct cblas_##name {}; \
  \
  template<> \
  struct cblas_##name<float> { \
    static BOOST_TYPEOF(cblas_##sname) *func; \
  }; \
  \
  template<> \
  struct cblas_##name<double> { \
    static BOOST_TYPEOF(cblas_##dname) *func; \
  }; \
} \

CBLAS_FUNC(copy, dcopy, scopy)
CBLAS_FUNC(scal, dscal, sscal)
CBLAS_FUNC(dot, ddot, sdot)
CBLAS_FUNC(iamax, idamax, isamax)
CBLAS_FUNC(axpy, daxpy, saxpy)
CBLAS_FUNC(gemv, dgemv, sgemv)
CBLAS_FUNC(symv, dsymv, ssymv)
CBLAS_FUNC(trmv, dtrmv, strmv)
CBLAS_FUNC(gbmv, dgbmv, sgbmv)
CBLAS_FUNC(gemm, dgemm, sgemm)
CBLAS_FUNC(symm, dsymm, ssymm)
CBLAS_FUNC(trmm, dtrmm, strmm)
CBLAS_FUNC(ger, dger, sger)
CBLAS_FUNC(syr, dsyr, ssyr)
CBLAS_FUNC(syr2, dsyr2, ssyr2)
CBLAS_FUNC(syrk, dsyrk, ssyrk)
CBLAS_FUNC(trsv, dtrsv, strsv)
CBLAS_FUNC(trsm, dtrsm, strsm)

#endif
