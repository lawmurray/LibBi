/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2422 $
 * $Date: 2012-03-22 14:10:45 +1100 (Thu, 22 Mar 2012) $
 */
#ifndef BI_CUDA_MATH_SIM_TEMP_VECTOR_HPP
#define BI_CUDA_MATH_SIM_TEMP_VECTOR_HPP

#include "temp_vector.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Temporary device vector type that is similar to some other given stencil
 * type.
 *
 * @ingroup math_matvec
 *
 * @tparam VM1 Vector or matrix type.
 *
 * sim_temp_gpu_vector is a convenience class for creating a temporary vector
 * on device that is similar to the given vector or matrix type @p VM1 in the
 * sense that it has the same scalar type.
 */
template<class VM1>
struct sim_temp_gpu_vector {
  /**
   * @internal
   */
  typedef typename VM1::value_type T;

  typedef temp_gpu_vector<T> type;
};
}

#endif
