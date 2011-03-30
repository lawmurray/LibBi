/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_LOCATABLE_HPP
#define BI_MATH_LOCATABLE_HPP

#include "host_vector.hpp"
#include "host_matrix.hpp"
#include "../cuda/math/vector.hpp"
#include "../cuda/math/matrix.hpp"
#include "../cuda/math/temp_vector.hpp"
#include "../cuda/math/temp_matrix.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Tags for location-specific data types and operations.
 */
enum Location {
  /**
   * Locate object on host.
   */
  ON_HOST = 0,

  /**
   * Locate object on device.
   */
  ON_DEVICE = 1
};

/**
 * Vector with location (host or device) given as template parameter.
 * Ultimately maps to host_vector or gpu_vector type.
 *
 * @ingroup math
 *
 * @tparam L Location.
 * @tparam T Value type.
 */
template<Location L, class T>
struct locatable_vector {
  typedef typename boost::mpl::if_c<L,
      gpu_vector<T>,
      host_vector<T> >::type type;
};

/**
 * Matrix with location (host or device) given as template parameter.
 * Ultimately maps to host_matrix or gpu_matrix type.
 *
 * @ingroup math
 *
 * @tparam L Location.
 * @tparam T Value type.
 */
template<Location L, class T>
struct locatable_matrix {
  typedef typename boost::mpl::if_c<L,
      gpu_matrix<T>,
      host_matrix<T> >::type type;
};

/**
 * Temporary vector with location (host or device) given as template
 * parameter.
 *
 * @ingroup math
 *
 * @tparam L Location.
 * @tparam T Value type.
 */
template<Location L, class T>
struct locatable_temp_vector {
  typedef typename boost::mpl::if_c<L,
      vector_temp_type<gpu_vector<T> >,
      vector_temp_type<host_vector<T> >
  >::type::type type;
};

/**
 * Temporary matrix with location (host or device) given as template
 * parameter.
 *
 * @ingroup math
 *
 * @tparam L Location.
 * @tparam T Value type.
 */
template<Location L, class T>
struct locatable_temp_matrix {
  typedef typename boost::mpl::if_c<L,
      matrix_temp_type<gpu_matrix<T> >,
      matrix_temp_type<host_matrix<T> >
  >::type::type type;
};

}

#endif
