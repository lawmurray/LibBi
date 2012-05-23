/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2422 $
 * $Date: 2012-03-22 14:10:45 +1100 (Thu, 22 Mar 2012) $
 */
#ifndef BI_MATH_LOC_VECTOR_HPP
#define BI_MATH_LOC_VECTOR_HPP

#include "vector.hpp"
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
 * loc_vector is a convenience class for creating a vector on host or device
 * according to a template argument.
 */
template<Location L, class T>
struct loc_vector {
  #ifdef ENABLE_GPU
  typedef typename boost::mpl::if_c<L,
      gpu_vector<T>,
      host_vector<T> >::type type;
  #else
  typedef host_vector<T> type;
  #endif
};
}

#endif
