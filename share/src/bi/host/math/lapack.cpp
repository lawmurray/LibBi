/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "lapack.hpp"

#define LAPACK_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(sname##_) *bi::lapack_##name<float>::func = sname##_; \
  BOOST_TYPEOF(dname##_) *bi::lapack_##name<double>::func = dname##_;

LAPACK_FUNC_DEF(potrf, dpotrf, spotrf)
LAPACK_FUNC_DEF(potrs, dpotrs, spotrs)
LAPACK_FUNC_DEF(syevx, dsyevx, ssyevx)
