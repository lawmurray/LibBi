/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_SIM_TEMP_MATRIX_HPP
#define BI_CUDA_MATH_SIM_TEMP_MATRIX_HPP

#include "temp_matrix.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Temporary device matrix type that is similar to some other given stencil
 * type.
 *
 * @ingroup math_matvec
 *
 * @tparam VM1 Vector or matrix type.
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 *
 * sim_temp_gpu_matrix is a convenience class for creating a temporary matrix
 * on device that is similar to the given vector or matrix type @p VM1 in the
 * sense that it has the same scalar type.
 */
template<class VM1, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = 1>
struct sim_temp_gpu_matrix {
  /**
   * @internal
   */
  typedef typename VM1::value_type T;

  typedef typename temp_gpu_matrix<T,size1_value,size2_value,lead_value,inc_value>::type type;
};
}

#endif
