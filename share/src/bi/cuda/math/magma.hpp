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

#ifdef HAVE_MAGMA_H
MAGMA_FUNC(potrf, dpotrf_gpu, spotrf_gpu)
MAGMA_FUNC(potrs, dpotrs_gpu, spotrs_gpu)
MAGMA_FUNC(get_potrf_nb, get_dpotrf_nb, get_spotrf_nb)
//MAGMA_FUNC(syevx, dsyevx_gpu, ssyevx_gpu)
#endif

#endif
