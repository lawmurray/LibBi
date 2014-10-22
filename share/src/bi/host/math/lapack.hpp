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
void ssyevx_(char* jobz, char* range, char* uplo, int* N, float* A, int* ldA,
    float* vl, float* vu, int* il, int* iu, float* abstol, int* m,
    float* w, float* Z, int* ldZ, float* work, int* lwork, int* iwork,
    int* ifail, int* info);
void dsyevx_(char* jobz, char* range, char* uplo, int* N, double* A, int* ldA,
    double* vl, double* vu, int* il, int* iu, double* abstol, int* m,
    double* w, double* Z, int* ldZ, double* work, int* lwork, int* iwork,
    int* ifail, int* info);
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
LAPACK_FUNC(syevx, dsyevx, ssyevx)

#endif
