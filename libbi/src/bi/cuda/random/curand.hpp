/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_CURAND_HPP
#define BI_RANDOM_CURAND_HPP

#include "curand.h"

#include "boost/typeof/typeof.hpp"

/**
 * @def CURAND_CHECKED_CALL(call)
 *
 * Make CURAND function call and assert success.
 *
 * @arg Function call.
 */
#if defined(__CUDACC__) && !defined(NDEBUG)
#define CURAND_CHECKED_CALL(call) \
  { \
    curandStatus err; \
    err = call; \
    BI_ASSERT(err == CURAND_STATUS_SUCCESS, "CURAND Error " << err); \
  }
#else
#define CURAND_CHECKED_CALL(call) call
#endif

/**
 * @def CURAND_FUNC(name, sname, dname)
 *
 * Macro for constructing template facades for CURAND functions.
 */
#define CURAND_FUNC(name, dname, sname) \
namespace bi { \
  template<class T> \
  struct curand_##name {}; \
  \
  template<> \
  struct curand_##name<float> { \
    static BOOST_TYPEOF(curand##sname) *func; \
  }; \
  \
  template<> \
  struct curand_##name<double> { \
    static BOOST_TYPEOF(curand##dname) *func; \
  }; \
}

CURAND_FUNC(generate_uniform, GenerateUniformDouble, GenerateUniform)
CURAND_FUNC(generate_normal, GenerateNormalDouble, GenerateNormal)

#endif
