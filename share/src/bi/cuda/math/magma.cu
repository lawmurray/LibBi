/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "magma.hpp"

#define MAGMA_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(magma_##sname) *bi::magma_##name<float>::func = magma_##sname; \
  BOOST_TYPEOF(magma_##dname) *bi::magma_##name<double>::func = magma_##dname;

MAGMA_FUNC_DEF(potrf, dpotrf_gpu, spotrf_gpu)
MAGMA_FUNC_DEF(potrs, dpotrs_gpu, spotrs_gpu)
MAGMA_FUNC_DEF(get_potrf_nb, get_dpotrf_nb, get_spotrf_nb)
//MAGMA_FUNC_DEF(syevx, dsyevx_gpu, ssyevx_gpu)
