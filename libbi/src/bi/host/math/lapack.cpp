/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1244 $
 * $Date: 2011-01-31 10:37:29 +0800 (Mon, 31 Jan 2011) $
 */
#include "lapack.hpp"

#define LAPACK_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(sname##_) *bi::lapack_##name<float>::func = sname##_; \
  BOOST_TYPEOF(dname##_) *bi::lapack_##name<double>::func = dname##_;

LAPACK_FUNC_DEF(potrf, dpotrf, spotrf)
LAPACK_FUNC_DEF(potrs, dpotrs, spotrs)
