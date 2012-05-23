/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2422 $
 * $Date: 2012-03-22 14:10:45 +1100 (Thu, 22 Mar 2012) $
 */
#ifndef BI_MATH_LOC_MATRIX_HPP
#define BI_MATH_LOC_MATRIX_HPP

#include "matrix.hpp"
#include "../misc/location.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Matrix with location designated by template parameter.
 *
 * @ingroup math_matvec
 *
 * @tparam L Location.
 * @tparam T Scalar type.
 *
 * loc_matrix is a convenience class for creating a matrix on host or device
 * according to a template argument.
 */
template<Location L, class T>
struct loc_matrix {
  #ifdef ENABLE_GPU
  typedef typename boost::mpl::if_c<L,
      gpu_matrix<T>,
      host_matrix<T> >::type type;
  #else
  typedef host_matrix<T> type;
  #endif
};
}

#endif
