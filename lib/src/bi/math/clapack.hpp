/**
 * @file
 *
 * Macros for CLAPACK usage.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_CLAPACK_HPP
#define BI_MATH_CLAPACK_HPP

extern "C" {
  #include "clapack.h"
}

#include "boost/typeof/typeof.hpp"

/**
 * @def CLAPACK_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for clapack functions.
 */
#define CLAPACK_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct clapack_##name {}; \
  \
  template<> \
  struct clapack_##name<float> { \
    static BOOST_TYPEOF(clapack_##sname) *func; \
  }; \
  \
  template<> \
  struct clapack_##name<double> { \
    static BOOST_TYPEOF(clapack_##dname) *func; \
  }; \
}

CLAPACK_FUNC(potrf, dpotrf, spotrf)
CLAPACK_FUNC(potrs, dpotrs, spotrs)

#endif
