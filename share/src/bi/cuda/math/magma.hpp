/**
 * @file
 *
 * Macros for MAGMA usage.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_MAGMA_HPP
#define BI_CUDA_MATH_MAGMA_HPP

#include "cublas.hpp"
#ifdef HAVE_MAGMA_H
#include "magma.h"
#else
int magma_dpotrf_gpu(char uplo, int n, double *A, int lda, int *info);
int magma_spotrf_gpu(char uplo, int n, float *A, int lda, int *info);

int magma_dpotrs_gpu(char uplo, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, int *info);
int magma_spotrs_gpu(char uplo, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, int *info);

int magma_get_dpotrf_nb(int m);
int magma_get_spotrf_nb(int m);
#endif

#include "boost/typeof/typeof.hpp"

/**
 * @def MAGMA_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for magma functions.
 */
#define MAGMA_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct magma_##name {}; \
  \
  template<> \
  struct magma_##name<float> { \
    static BOOST_TYPEOF(magma_##sname) *func; \
  }; \
  \
  template<> \
  struct magma_##name<double> { \
    static BOOST_TYPEOF(magma_##dname) *func; \
  }; \
}

MAGMA_FUNC(potrf, dpotrf_gpu, spotrf_gpu)
MAGMA_FUNC(potrs, dpotrs_gpu, spotrs_gpu)
MAGMA_FUNC(get_potrf_nb, get_dpotrf_nb, get_spotrf_nb)
//MAGMA_FUNC(syevx, dsyevx_gpu, ssyevx_gpu)

#endif
