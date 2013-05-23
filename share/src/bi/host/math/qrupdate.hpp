/**
 * @file
 *
 * Function declarations and macros for qrupdate usage. See Fortran source
 * of qrupdate for function documentation.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_QRUPDATE_HPP
#define BI_HOST_MATH_QRUPDATE_HPP

extern "C" {
  void sch1up_(int* n, float* R, int& ldr, float* u, float* w);
  void dch1up_(int* n, double* R, int& ldr, double* u, double* w);
  void sch1dn_(int* n, float* R, int* ldr, float* u, float* w, int* info);
  void dch1dn_(int* n, double* R, int* ldr, double* u, double* w, int* info);
}

#include "boost/typeof/typeof.hpp"

/**
 * @def QRUPDATE_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for qrupdate functions.
 */
#define QRUPDATE_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct qrupdate_##name {}; \
  \
  template<> \
  struct qrupdate_##name<float> { \
    static BOOST_TYPEOF(sname##_) *func; \
  }; \
  \
  template<> \
  struct qrupdate_##name<double> { \
    static BOOST_TYPEOF(dname##_) *func; \
  }; \
}

QRUPDATE_FUNC(ch1up, dch1up, sch1up)
QRUPDATE_FUNC(ch1dn, dch1dn, sch1dn)

#endif
