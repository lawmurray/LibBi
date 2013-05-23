/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_LOC_TEMP_VECTOR_HPP
#define BI_MATH_LOC_TEMP_VECTOR_HPP

#include "temp_vector.hpp"
#include "../misc/location.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Vector with location designated by template parameter.
 *
 * @ingroup math_matvec
 *
 * @tparam L Location.
 * @tparam T Scalar type.
 * @tparam size_value Static size, -1 for dynamic.
 * @tparam inc_value Static increment, -1 for dynamic.
 *
 * loc_temp_vector is a convenience class for creating a temporary vector on
 * host or device according to a template argument.
 */
template<Location L, class T, int size_value = -1, int inc_value = 1>
struct loc_temp_vector {
  #ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<L,
      temp_gpu_vector<T,size_value,inc_value>,
      temp_host_vector<T,size_value,inc_value> >::type::type type;
  #else
  typedef typename temp_host_vector<T,size_value,inc_value>::type type;
  #endif
};
}

#endif
