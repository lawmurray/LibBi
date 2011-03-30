/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1253 $
 * $Date: 2011-02-01 12:03:44 +0800 (Tue, 01 Feb 2011) $
 */
#include "curand.hpp"

#define CURAND_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(curand##sname) *bi::curand_##name<float>::func = curand##sname; \
  BOOST_TYPEOF(curand##dname) *bi::curand_##name<double>::func = curand##dname;

CURAND_FUNC_DEF(generate_uniform, GenerateUniformDouble, GenerateUniform)
CURAND_FUNC_DEF(generate_normal, GenerateNormalDouble, GenerateNormal)
