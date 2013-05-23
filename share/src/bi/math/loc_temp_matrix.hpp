/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_LOC_TEMP_MATRIX_HPP
#define BI_MATH_LOC_TEMP_MATRIX_HPP

#include "temp_matrix.hpp"
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
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 *
 * loc_temp_matrix is a convenience class for creating a temporary matrix on
 * host or device according to a template argument.
 */
template<Location L, class T, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = 1>
struct loc_temp_matrix {
  #ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<L,
      temp_gpu_matrix<T,size1_value,size2_value,lead_value,inc_value>,
      temp_host_matrix<T,size1_value,size2_value,lead_value,inc_value> >::type::type type;
  #else
  typedef typename temp_host_matrix<T,size1_value,size2_value,lead_value,inc_value>::type type;
  #endif
};
}

#endif
