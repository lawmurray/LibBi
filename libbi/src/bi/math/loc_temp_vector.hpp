/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2422 $
 * $Date: 2012-03-22 14:10:45 +1100 (Thu, 22 Mar 2012) $
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
 *
 * loc_temp_vector is a convenience class for creating a temporary vector on
 * host or device according to a template argument.
 */
template<Location L, class T>
struct loc_temp_vector {
  #ifdef ENABLE_GPU
  typedef typename boost::mpl::if_c<L,
      temp_gpu_vector<T>,
      temp_host_vector<T> >::type::type type;
  #else
  typedef typename temp_host_vector<T>::type type;
  #endif
};
}

#endif
