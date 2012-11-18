/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_SIM_TEMP_VECTOR_HPP
#define BI_MATH_SIM_TEMP_VECTOR_HPP

#include "temp_vector.hpp"
#include "../host/math/sim_temp_vector.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/math/sim_temp_vector.hpp"
#endif

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Temporary vector type that is similar to some other given stencil type.
 *
 * @ingroup math_matvec
 *
 * @tparam VM1 Vector or matrix type.
 * @tparam size_value Static size, -1 for dynamic.
 * @tparam inc_value Static increment, -1 for dynamic.
 *
 * sim_temp_vector is a convenience class for creating a temporary vector
 * that is similar to the given vector or matrix type @p VM1 in the sense
 * that:
 *
 * @li it resides in the same location (host or device), and
 * @li it has the same scalar type.
 */
template<class VM1, int size_value = -1, int inc_value = 1>
struct sim_temp_vector {
  /**
   * @internal
   */
  typedef typename VM1::value_type T;

  #ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<VM1::on_device,
      temp_gpu_vector<T,size_value,inc_value>,
      temp_host_vector<T,size_value,inc_value> >::type::type type;
  #else
  typedef typename temp_host_vector<T,size_value,inc_value>::type type;
  #endif
};
}

#endif
