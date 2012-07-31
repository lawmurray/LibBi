/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_SIM_TEMP_MATRIX_HPP
#define BI_MATH_SIM_TEMP_MATRIX_HPP

#include "temp_matrix.hpp"
#include "../host/math/sim_temp_matrix.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/math/sim_temp_matrix.hpp"
#endif

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Temporary matrix type that is similar to some other given stencil type.
 *
 * @ingroup math_matvec
 *
 * @tparam VM1 Vector or matrix type.
 *
 * sim_temp_matrix is a convenience class for creating a temporary matrix
 * that is similar to the given vector or matrix type @p VM1 in the sense
 * that:
 *
 * @li it resides in the same location (host or device), and
 * @li it has the same scalar type.
 */
template<class VM1>
struct sim_temp_matrix {
  /**
   * @internal
   */
  typedef typename VM1::value_type T;

  #ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<VM1::on_device,
      temp_gpu_matrix<T>,
      temp_host_matrix<T> >::type::type type;
  #else
  typedef typename temp_host_matrix<T>::type type;
  #endif
};
}

#endif
