/**
 * @file
 *
 * Macros for LAPACK usage.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_LAPACK_HPP
#define BI_HOST_MATH_LAPACK_HPP

extern "C" {
  void spotrf_(char* uplo, int* n, float* A, int* lda, int* info);
  void dpotrf_(char* uplo, int* n, double* A, int* lda, int* info);
  void spotrs_(char* uplo, int* n, int* nhrs, float* A, int* lda, float* B,
      int* ldb, int* info);
  void dpotrs_(char* uplo, int* n, int* nhrs, double* A, int* lda, double* B,
      int* ldb, int* info);
}

#include "boost/typeof/typeof.hpp"

/**
 * @defCLAPACK_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for LAPACK functions.
 */
#define LAPACK_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct lapack_##name {}; \
  \
  template<> \
  struct lapack_##name<float> { \
    static BOOST_TYPEOF(sname##_) *func; \
  }; \
  \
  template<> \
  struct lapack_##name<double> { \
    static BOOST_TYPEOF(dname##_) *func; \
  }; \
}

LAPACK_FUNC(potrf, dpotrf, spotrf)
LAPACK_FUNC(potrs, dpotrs, spotrs)

#endif
