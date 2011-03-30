/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "clapack.hpp"

#define CLAPACK_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(clapack_##sname) *bi::clapack_##name<float>::func = clapack_##sname; \
  BOOST_TYPEOF(clapack_##dname) *bi::clapack_##name<double>::func = clapack_##dname;

CLAPACK_FUNC_DEF(potrf, dpotrf, spotrf)
CLAPACK_FUNC_DEF(potrs, dpotrs, spotrs)
